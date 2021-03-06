#include "dsaproblem.h"

#include "angular_quadrature.h"
#include "description.h"
#include "discretization.h"
#include "problem.h"

#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
DSAProblem<dim>::DSAProblem(Problem<dim> & problem)
  : ParameterAcceptor("DSAProblem"),
    comm(problem.get_comm()),
    pcout(problem.get_pcout()),
    timer(problem.get_timer()),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old()),
    reflective_dof_normals(problem.get_reflective_dof_normals()),
    reflective_incoming_flux(problem.get_reflective_incoming_flux()),
    reflective_dJ(problem.get_reflective_dJ()),
    system_matrix(problem.get_system_matrix()),
    system_rhs(problem.get_system_rhs()),
    system_solution(problem.get_system_solution())
{
  // Whether or not reflective bc acceleration is enabled (default: true)
  add_parameter("reflective_bc_acceleration", reflective_bc_acceleration);

  // Factor to multiply the penalty coefficient (default: 1)
  add_parameter("kappa_c_factor", kappa_c_factor);

  // Whether or not to enable detailed solver output (default: false)
  add_parameter("detailed_solver_output", detailed_solver_output);
  // Relative tolerance (default: 1e-12)
  add_parameter("relative_tolerance", relative_tolerance);
  // Absolute tolerance (default: 1e-12)
  add_parameter("absolute_tolerance", absolute_tolerance);
}

template <int dim>
void
DSAProblem<dim>::setup()
{
  TimerOutput::Scope t(timer, "DSAProblem setup");

  // Warn if kappa_c_factor < 1
  if (kappa_c_factor < 1)
    pcout << "Warning: kappa_c_factor in DSAProblem should likely be >= 1!\n\n";

  // Initialize constant system storage
  dsa_matrix.reinit(discretization.get_locally_owned_dofs(),
                    discretization.get_locally_owned_dofs(),
                    discretization.get_sparsity_pattern(),
                    comm);

  // Assemble the constant LHS
  assemble_initial();
}

template <int dim>
void
DSAProblem<dim>::assemble_solve_update()
{
  // Assemble the parts of the LHS and RHS that are changing
  assemble();

  // And solve
  solve();

  // Update scalar flux with change. Note that we do not update the angular flux
  // if enabled beacuse by the time we reach convergence, DSA contribution will
  // be insignificant
  scalar_flux += system_solution;

  // Update incoming angular fluxes on reflecting boundaries with change
  if (description.has_reflecting_bcs())
  {
    TimerOutput::Scope t(timer, "DSAProblem reflective BC update");
    for (unsigned int d = 0; d < aq.n_dir(); ++d)
    {
      const auto dir = aq.dir(d);
      // Loop through each entry for the incoming angular flux for direction d
      // and update with the DSA correction
      for (auto & dof_value_pair : reflective_incoming_flux[d])
      {
        const unsigned int dof = dof_value_pair.first;
        double & value = dof_value_pair.second;
        const double omega_dot_n = dir * get_hat_direction<dim>(reflective_dof_normals.at(dof));
        value += system_solution[dof] - reflective_dJ.at(dof) * omega_dot_n;
      }
    }
  }
}

template <int dim>
void
DSAProblem<dim>::solve()
{
  TimerOutput::Scope t1(timer, "DSAProblem solve");

  system_solution = 0;
  SolverControl control(1000, relative_tolerance * system_rhs.l2_norm() + absolute_tolerance);
  LA::SolverCG::AdditionalData cg_data(detailed_solver_output);
  LA::SolverCG solver(control, cg_data);
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix);
  solver.solve(system_matrix, system_solution, system_rhs, preconditioner);

  pcout << "  DSA converged after " << control.last_step() << " CG iterations" << std::endl;
}

template <int dim>
void
DSAProblem<dim>::assemble_initial()
{
  // Need shape function values, JxW, and shape function gradieints on cells and faces
  UpdateFlags update_flags = update_values | update_JxW_values | update_gradients;
  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(dof_handler.get_fe(), discretization.get_mapping());

  // Initialize the assembler for a simple system with dsa_matrix; system_rhs
  // is used in initilization but is never updated in the initial setup
  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector> assembler;
  assembler.initialize(dsa_matrix, system_rhs);

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info) {
    DSAProblem::integrate_cell_initial(dinfo, info);
  };
  const auto boundary_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info) {
    DSAProblem::integrate_boundary_initial(dinfo, info);
  };
  const auto face_worker = [&](MeshWorker::DoFInfo<dim> & dinfo1,
                               MeshWorker::DoFInfo<dim> & dinfo2,
                               MeshWorker::IntegrationInfo<dim> & info1,
                               MeshWorker::IntegrationInfo<dim> & info2) {
    DSAProblem::integrate_face_initial(dinfo1, dinfo2, info1, info2);
  };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<dim> dof_info(dof_handler);
  MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      cell_worker,
      boundary_worker,
      face_worker,
      assembler);

  // Bring all the matrices together
  dsa_matrix.compress(VectorOperation::add);
}

template <int dim>
void
DSAProblem<dim>::assemble()
{
  TimerOutput::Scope t(timer, "DSAProblem assembly");

  // Need shape function values and JxW on cells
  UpdateFlags update_flags = update_values | update_JxW_values;
  MeshWorker::IntegrationInfoBox<dim> info_box;
  info_box.add_update_flags_cell(update_flags);
  // Also need shape function values and JxW on boundary faces w/ reflecting bcs
  if (description.has_reflecting_bcs() && reflective_bc_acceleration)
    info_box.add_update_flags_boundary(update_flags);
  info_box.initialize(dof_handler.get_fe(), discretization.get_mapping());

  // Initialize the assembler for a symple system with system_matrix and system_rhs
  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector> assembler;
  assembler.initialize(system_matrix, system_rhs);

  // Copy over LHS and reset RHS
  system_matrix.copy_from(dsa_matrix);
  system_rhs = 0;

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info) {
    DSAProblem::integrate_cell(dinfo, info);
  };
  const auto boundary_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info) {
    DSAProblem::integrate_boundary(dinfo, info);
  };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<dim> dof_info(dof_handler);
  MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      cell_worker,
      boundary_worker,
      NULL,
      assembler);
}

template <int dim>
void
DSAProblem<dim>::integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                                MeshWorker::IntegrationInfo<dim> & info) const
{
  const FEValuesBase<dim> & fe = info.fe_values();
  Vector<double> & local_vector = dinfo.vector(0).block(0);

  // Scattering cross section for this cell
  const double sigma_s = description.get_material(dinfo.cell->material_id()).sigma_s;

  // Change in scalar flux at each vertex for the scattering "source"
  std::vector<double> scalar_flux_change(fe.n_quadrature_points, 0);
  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
  {
    const double change = scalar_flux(dinfo.indices[i]) - scalar_flux_old(dinfo.indices[i]);
    for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
      scalar_flux_change[q] += fe.shape_value(i, q) * change;
  }

  // Integration of sigma_s * FEM representation of the change in the scalar flux
  // in the cell
  LocalIntegrators::L2::L2(local_vector, fe, scalar_flux_change, sigma_s);
}

template <int dim>
void
DSAProblem<dim>::integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                                    MeshWorker::IntegrationInfo<dim> & info) const
{
  // Nothing to do without reflective bc correction
  if (!reflective_bc_acceleration || !description.has_reflecting_bcs() ||
      description.get_bc(dinfo.face->boundary_id()).type != ReflectiveBC)
    return;

  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const auto & fe = dof_handler.get_fe();
  const FEValuesBase<dim> & fe_v = info.fe_values();
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Integration of the FEM representation of dJ on the boundary if it is a
  // reflecting boundary only
  for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    if (fe.has_support_on_face(i, dinfo.face_number))
    {
      const double dJ_i = reflective_dJ.at(dinfo.indices[i]);
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        local_vector(i) += fe_v.shape_value(i, q) * JxW[q] * dJ_i;
    }
}

template <int dim>
void
DSAProblem<dim>::integrate_cell_initial(MeshWorker::DoFInfo<dim> & dinfo,
                                        MeshWorker::IntegrationInfo<dim> & info) const
{
  const FEValuesBase<dim> & fe = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  const auto & material = description.get_material(dinfo.cell->material_id());

  // Mass matrix * sigma_a on each cell
  LocalIntegrators::L2::mass_matrix(local_matrix, fe, material.sigma_a);
  // D * the Laplacian on each cell
  LocalIntegrators::Laplace::cell_matrix(local_matrix, fe, material.D);
}

template <int dim>
void
DSAProblem<dim>::integrate_face_initial(MeshWorker::DoFInfo<dim> & dinfo1,
                                        MeshWorker::DoFInfo<dim> & dinfo2,
                                        MeshWorker::IntegrationInfo<dim> & info1,
                                        MeshWorker::IntegrationInfo<dim> & info2) const
{
  // FEValues for each cell
  const auto & fe1 = info1.fe_values();
  const auto & fe2 = info2.fe_values();

  // Diffusion coefficient in each cell
  const double D1 = description.get_material(dinfo1.cell->material_id()).D;
  const double D2 = description.get_material(dinfo2.cell->material_id()).D;

  // Length of the cells in the orthogonal direction to this face
  const unsigned int n1 = GeometryInfo<dim>::unit_normal_direction[dinfo1.face_number];
  const unsigned int n2 = GeometryInfo<dim>::unit_normal_direction[dinfo2.face_number];
  const double h1 = dinfo1.cell->extent_in_direction(n1);
  const double h2 = dinfo2.cell->extent_in_direction(n2);
  // Polynomial degrees on each face
  const unsigned int deg1 = fe1.get_fe().tensor_degree();
  const unsigned int deg2 = fe2.get_fe().tensor_degree();
  // Penalty coefficient
  const double c1 = kappa_c_factor * 4 * deg1 * (deg1 + 1);
  const double c2 = kappa_c_factor * 4 * deg2 * (deg2 + 1);
  const double kappa = std::fmax(0.5 * (c1 * D1 / h1 + c2 * D2 / h2), 0.25);

  // The following is very similar to LocalIntegrators::Laplace::ip_matrix, with
  // subtle changes in regards to the penalty coefficient
  FullMatrix<double> & v1u1 = dinfo1.matrix(0, false).matrix;
  FullMatrix<double> & v1u2 = dinfo1.matrix(0, true).matrix;
  FullMatrix<double> & v2u1 = dinfo2.matrix(0, true).matrix;
  FullMatrix<double> & v2u2 = dinfo2.matrix(0, false).matrix;
  const unsigned int n_dofs = fe1.dofs_per_cell;
  for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
  {
    const double JxW = fe1.JxW(q);
    const Tensor<1, dim> n = fe1.normal_vector(q);
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const double Ddnv1 = D1 * n * fe1.shape_grad(i, q);
      const double Ddnv2 = D2 * n * fe2.shape_grad(i, q);
      const double v1 = fe1.shape_value(i, q);
      const double v2 = fe2.shape_value(i, q);
      for (unsigned int j = 0; j < n_dofs; ++j)
      {
        const double Ddnu1 = D1 * n * fe1.shape_grad(j, q);
        const double Ddnu2 = D2 * n * fe2.shape_grad(j, q);
        const double u1 = fe1.shape_value(j, q);
        const double u2 = fe2.shape_value(j, q);
        v1u1(i, j) += JxW * (+kappa * v1 * u1 + 0.5 * (+v1 * Ddnu1 + Ddnv1 * u1));
        v1u2(i, j) += JxW * (-kappa * v1 * u2 + 0.5 * (+v1 * Ddnu2 - Ddnv1 * u2));
        v2u1(i, j) += JxW * (-kappa * v2 * u1 + 0.5 * (-v2 * Ddnu1 + Ddnv2 * u1));
        v2u2(i, j) += JxW * (+kappa * v2 * u2 + 0.5 * (-v2 * Ddnu2 - Ddnv2 * u2));
      }
    }
  }
}

template <int dim>
void
DSAProblem<dim>::integrate_boundary_initial(MeshWorker::DoFInfo<dim> & dinfo,
                                            MeshWorker::IntegrationInfo<dim> & info) const
{
  const auto & fe = info.fe_values();

  // Bounday condition type for this face
  const auto bc_type = description.get_bc(dinfo.face->boundary_id()).type;

  // Diffusion coefficient
  const double D = description.get_material(dinfo.cell->material_id()).D;

  // RHS contribution (for dirichlet BCs)
  if (bc_type != ReflectiveBC)
  {
    // Length of the cell in the orthogonal direction to this face
    const unsigned int n = GeometryInfo<dim>::unit_normal_direction[dinfo.face_number];
    const double h = dinfo.cell->extent_in_direction(n);
    // Polynomial degrees on the face
    const unsigned int deg = info.fe_values().get_fe().tensor_degree();
    // Penalty coefficient
    const double c = kappa_c_factor * 4 * deg * (deg + 1);
    const double kappa = std::fmax(c * D / h, 0.25);

    // "factor" used in nitsche_matrix multiplies the entire term, therefore we
    // must divide the penalty factor by the same factor (which is D)
    LocalIntegrators::Laplace::nitsche_matrix(dinfo.matrix(0).matrix, fe, kappa / D, D);
  }
}

template DSAProblem<2>::DSAProblem(Problem<2> & problem);
template DSAProblem<3>::DSAProblem(Problem<3> & problem);

template void DSAProblem<2>::setup();
template void DSAProblem<3>::setup();

template void DSAProblem<2>::assemble_solve_update();
template void DSAProblem<3>::assemble_solve_update();

} // namespace RadProblem
