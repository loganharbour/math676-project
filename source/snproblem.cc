#include "snproblem.h"

#include "angular_quadrature.h"
#include "description.h"
#include "discretization.h"
#include "problem.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
SNProblem<dim>::SNProblem(Problem<dim> & problem)
  : ParameterAcceptor("SNProblem"),
    comm(problem.get_comm()),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old()),
    system_matrix(problem.get_system_matrix()),
    system_rhs(problem.get_system_rhs()),
    system_solution(problem.get_system_solution())
{
}

template <int dim>
void
SNProblem<dim>::setup()
{
  // Setup InfoBox for MeshWorker
  const unsigned int n_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_points, n_points, n_points);
  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.initialize(dof_handler.get_fe(), discretization.get_mapping());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(system_matrix, system_rhs);
}

template <int dim>
void
SNProblem<dim>::solve_directions()
{
  // Copy to old scalar flux and zero scalar flux for update
  scalar_flux_old = scalar_flux;
  scalar_flux = 0;

  // Solve each direction
  for (unsigned int d = 0; d < aq.n_dir(); ++d)
    solve_direction(d);
}

template <int dim>
void
SNProblem<dim>::solve_direction(const unsigned int d)
{
  // Assemble the system
  assemble_direction(aq.dir(d));

  system_solution = 0;

  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG solver(solver_control);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  preconditioner.initialize(system_matrix, data);
  solver.solve(system_matrix, system_solution, system_rhs, preconditioner);

  std::cout << "  Direction " << d << " converged after " << solver_control.last_step()
            << " iterations " << std::endl;

  // Update scalar flux at each node (weighed by angular weight)
  system_solution *= aq.w(d);
  scalar_flux += system_solution;
}

template <int dim>
void
SNProblem<dim>::assemble_direction(const Tensor<1, dim> & dir)
{
  // Zero lhs and rhs before assembly
  system_matrix = 0;
  system_rhs = 0;

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info) {
    SNProblem::integrate_cell(dinfo, info, dir);
  };
  const auto boundary_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info) {
    SNProblem::integrate_boundary(dinfo, info, dir);
  };
  const auto face_worker = [&](MeshWorker::DoFInfo<dim> & dinfo1,
                               MeshWorker::DoFInfo<dim> & dinfo2,
                               MeshWorker::IntegrationInfo<dim> & info1,
                               MeshWorker::IntegrationInfo<dim> & info2) {
    SNProblem::integrate_face(dinfo1, dinfo2, info1, info2, dir);
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
}

template <int dim>
void
SNProblem<dim>::integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info,
                               const Tensor<1, dim> & dir) const
{
  const FEValuesBase<dim> & fe_v = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Material for this cell
  const auto & material = description.get_material(dinfo.cell->material_id());

  // Whether or not this cell has scattering
  const bool has_scattering = material.sigma_s != 0;

  // Compute old scalar flux at each quadrature point for scattering source
  std::vector<double> scattering_source;
  if (has_scattering)
  {
    double scalar_flux_old_i;
    scattering_source.resize(fe_v.n_quadrature_points);
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      scalar_flux_old_i = scalar_flux_old[dinfo.indices[i]];
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        scattering_source[q] += material.sigma_s * scalar_flux_old_i * fe_v.shape_value(i, q);
    }
  }

  for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      const double v_i = fe_v.shape_value(i, q);
      const double dir_dot_grad_v_i = dir * fe_v.shape_grad(i, q);
      for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
        // Streaming + collision
        local_matrix(i, j) +=
            (fe_v.shape_value(j, q) * JxW[q]) * (material.sigma_t * v_i - dir_dot_grad_v_i);
      // External source
      local_vector(i) += v_i * material.src * JxW[q];
      // Scattering source
      if (has_scattering)
        local_vector(i) += v_i * scattering_source[q] * JxW[q];
    }
}

template <int dim>
void
SNProblem<dim>::integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info,
                                   const Tensor<1, dim> & dir) const
{
  // Dot product between the direction and the outgoing normal
  const double dir_dot_n = dir * info.fe_values().normal_vector(0);

  // Face is incoming
  if (dir_dot_n < 0)
  {
    // Nothing to do for incoming without incident boundary conditions
    if (!description.has_incident_bcs())
      return;

    const auto & bc = description.get_bc(dinfo.face->boundary_id());

    double bc_value = 0;
    switch (bc.type)
    {
      case BCTypes::Vacuum:
        return;
      case BCTypes::Isotropic:
        bc_value = bc.value;
        break;
      case BCTypes::Reflective:
        throw ExcMessage("Reflective bc not supported yet");
      case BCTypes::Perpendicular:
        throw ExcMessage("Perpendicular bc not supported yet");
    }

    const FEValuesBase<dim> & fe_v = info.fe_values();
    Vector<double> & local_vector = dinfo.vector(0).block(0);
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        local_vector(i) += -bc_value * dir_dot_n * fe_v.shape_value(i, q) * JxW[q];
  }
  // Face is outgoing
  else
  {
    const FEValuesBase<dim> & fe_v = info.fe_values();
    FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
      {
        const double coeff = dir_dot_n * fe_v.shape_value(i, q) * JxW[q];
        for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
          local_matrix(i, j) += coeff * fe_v.shape_value(j, q);
      }
  }
}

template <int dim>
void
SNProblem<dim>::integrate_face(MeshWorker::DoFInfo<dim> & dinfo1,
                               MeshWorker::DoFInfo<dim> & dinfo2,
                               MeshWorker::IntegrationInfo<dim> & info1,
                               MeshWorker::IntegrationInfo<dim> & info2,
                               const Tensor<1, dim> & dir) const
{
  // Dot product between the direction and the outgoing normal of face 1
  const double dir_dot_n1 = dir * info1.fe_values().normal_vector(0);

  // Whether the first cell is the outgoing cell or not
  const bool cell1_out = dir_dot_n1 > 0;

  // FE values access for the outgoing and incoming cell
  const auto & fe_v_out = (cell1_out ? info1.fe_values() : info2.fe_values());
  const auto & fe_v_in = (cell1_out ? info2.fe_values() : info1.fe_values());

  // System matrices for the u_out v_out matrix and the u_out v_in matrix
  auto & uout_vout_matrix = (cell1_out ? dinfo1 : dinfo2).matrix(0, false).matrix;
  auto & uout_vin_matrix = (cell1_out ? dinfo2 : dinfo1).matrix(0, true).matrix;

  // Reverse the direction for the dot product if cell 2 is the outgoing cell
  const double dir_dot_nout = (cell1_out ? dir_dot_n1 : -dir_dot_n1);

  // Use quadrature points from cell 1; reference element is the same
  const std::vector<double> & JxW = info1.fe_values().get_JxW_values();
  for (unsigned int q = 0; q < fe_v_out.n_quadrature_points; ++q)
    for (unsigned int j = 0; j < fe_v_out.dofs_per_cell; ++j)
    {
      const double coeff = dir_dot_nout * fe_v_out.shape_value(j, q) * JxW[q];
      for (unsigned int i = 0; i < fe_v_out.dofs_per_cell; ++i)
        uout_vout_matrix(i, j) += coeff * fe_v_out.shape_value(i, q);
      for (unsigned int i = 0; i < fe_v_in.dofs_per_cell; ++i)
        uout_vin_matrix(i, j) -= coeff * fe_v_in.shape_value(i, q);
    }
}

template SNProblem<1>::SNProblem(Problem<1> & problem);
template SNProblem<2>::SNProblem(Problem<2> & problem);
template SNProblem<3>::SNProblem(Problem<3> & problem);

template void SNProblem<1>::setup();
template void SNProblem<2>::setup();
template void SNProblem<3>::setup();

template void SNProblem<1>::solve_directions();
template void SNProblem<2>::solve_directions();
template void SNProblem<3>::solve_directions();

} // namespace RadProblem
