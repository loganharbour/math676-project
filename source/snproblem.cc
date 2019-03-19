#include "snproblem.h"

#include "problem.h"

#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>

namespace RadProblem
{
using namespace dealii;

SNProblem::SNProblem(Problem & problem)
  : ParameterAcceptor("SNProblem"),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old())
{
  // Source iteration tolerance (defaut: 1e-12)
  add_parameter("source_iteration_tolerance", source_iteration_tolerance);
}

void
SNProblem::setup()
{
  // Initialize system storage for a single direction
  rhs.reinit(dof_handler.n_dofs());
  matrix.reinit(discretization.get_sparsity_pattern());
  solution.reinit(dof_handler.n_dofs());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(matrix, rhs);
}

bool
SNProblem::solve_directions(const unsigned int l)
{
  // Copy to old scalar flux and zero scalar flux for update
  scalar_flux_old = scalar_flux;
  scalar_flux = 0;

  // Solve each direction
  for (unsigned int d = 0; d < aq.n_dir(); ++d)
    solve_direction(d);

  // Source iteration: check for convergence
  if (description.has_scattering())
  {
    const double norm = L2_difference(scalar_flux, scalar_flux_old);
    residuals.push_back(norm);
    std::cout << "Source iteration " << l << " norm: " << std::scientific << std::setprecision(2)
              << norm << std::endl
              << std::endl;

    // Return true if converged
    if (norm < source_iteration_tolerance)
      return true;
    // Return false if not converged
    else
      return false;
  }
  // No scattering: return true for converged
  else
    return true;
}

void
SNProblem::solve_direction(const unsigned int d)
{
  std::cout << "Solving direction " << d;

  // See if renumbering is required
  const unsigned int half = (d < aq.n_dir() / 2 ? 0 : 1);
  const bool renumber_flux = (discretization.do_renumber() && half == 0 ? true : false);

  // Renumber dofs at the beginning of a half range if enabled
  if (discretization.do_renumber() && (d == 0 || d == aq.n_dir() / 2))
    discretization.renumber_dofs(half);

  // Assemble the system
  assemble_direction(aq.dir(d), renumber_flux);

  // Solve the system
  if (discretization.do_renumber())
    solve_gauss_seidel();
  else
    solve_richardson();

  // Update scalar flux at each node (weighed by angular weight)
  update_scalar_flux(aq.w(d), renumber_flux);
}

void
SNProblem::assemble_direction(const Tensor<1, 2> & dir, const bool renumber_flux)
{
  // Zero lhs and rhs before assembly
  matrix = 0;
  rhs = 0;

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    SNProblem::integrate_cell(dinfo, info, dir, renumber_flux);
  };
  const auto boundary_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    SNProblem::integrate_boundary(dinfo, info, dir);
  };
  const auto face_worker =
      [&](DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) {
        SNProblem::integrate_face(dinfo1, dinfo2, info1, info2, dir);
      };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<2> dof_info(dof_handler);
  MeshWorker::loop<2, 2, MeshWorker::DoFInfo<2>, MeshWorker::IntegrationInfoBox<2>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      discretization.info_box,
      cell_worker,
      boundary_worker,
      face_worker,
      assembler);
}

void
SNProblem::integrate_cell(DoFInfo & dinfo,
                          CellInfo & info,
                          const Tensor<1, 2> dir,
                          const bool renumber_flux)
{
  const FEValuesBase<2> & fe_v = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Get material for this cell
  auto search = materials.find(dinfo.cell->material_id());
  Assert(search != materials.end(), ExcMessage("Material id not found in material map"));
  const Material & material = search->second;

  // Whether or not this cell has scattering
  const bool has_scattering = material.sigma_s != 0;

  // Compute old scalar flux at each quadrature point for scattering source
  std::vector<double> scattering_source;
  if (has_scattering)
  {
    double scalar_flux_old_i;
    unsigned int index;
    scattering_source.resize(fe_v.n_quadrature_points);
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      if (renumber_flux)
        index = discretization.get_ref_renumbering(dinfo.indices[i]);
      else
        index = dinfo.indices[i];
      scalar_flux_old_i = scalar_flux_old[index];
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

void
SNProblem::integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Tensor<1, 2> dir)
{
  // Dot product between the direction and the outgoing normal
  const double dir_dot_n = dir * info.fe_values().normal_vector(0);

  // Face is incoming: check incident boundary conditions
  if (dir_dot_n < 0)
  {
    // Exit face if problem does not have incident boundary conditions
    if (!description.has_incident_bcs())
      return;

    double bc_value = 0;
    // Search for an isotropic boundary condition
    const auto & isotropic_bcs = description.get_isotropic_bcs();
    auto search = isotropic_bcs.find(dinfo.face->boundary_id());
    // Found an isotropic boundary condition
    if (search != isotropic_bcs.end())
      bc_value = search->second;
    // Search for a perpendicular boundary condition
    else
    {
      const auto & perpendicular_bcs = description.get_perpendicular_bcs();
      search = perpendicular_bcs.find(dinfo.face->boundary_id());
      // Found a perpendicular boundary condition
      if (search != perpendicular_bcs.end())
        bc_value = search->second;
      // Didn't find any incident boundary conditions
      else
        return;
    }

    // Continue if we have a nonzero value
    const FEValuesBase<2> & fe_v = info.fe_values();
    Vector<double> & local_vector = dinfo.vector(0).block(0);
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        local_vector(i) += -bc_value * dir_dot_n * fe_v.shape_value(i, q) * JxW[q];
  }
  // Face is outgoing
  else
  {
    const FEValuesBase<2> & fe_v = info.fe_values();
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

void
SNProblem::integrate_face(
    DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2, const Tensor<1, 2> dir)
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

void
SNProblem::solve_richardson()
{
  SolverControl solver_control(1000, 1e-12);
  SolverRichardson<> solver(solver_control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(matrix, discretization.get_fe().dofs_per_cell);
  solver.solve(matrix, solution, rhs, preconditioner);
  std::cout << " - converged after " << solver_control.last_step() << " iterations " << std::endl;
}

void
SNProblem::solve_gauss_seidel()
{
  PreconditionBlockSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(matrix, discretization.get_fe().dofs_per_cell);
  preconditioner.step(solution, rhs);
  std::cout << std::endl;
}

void
SNProblem::update_scalar_flux(const double weight, const bool renumber_flux)
{
  if (renumber_flux)
  {
    const auto & to_ref = discretization.get_ref_renumbering();
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      scalar_flux[i] += weight * solution[to_ref[i]];
  }
  else
    scalar_flux.add(weight, solution);
}

double
SNProblem::L2_difference(const Vector<double> & v1, const Vector<double> & v2)
{
  double value = 0;

  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    const FEValuesBase<2> & fe_v = info.fe_values();
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      const double diff_i = v1[dinfo.indices[i]] - v2[dinfo.indices[i]];
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        value += std::pow(fe_v.shape_value(i, q) * diff_i, 2) * JxW[q];
    }
  };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<2> dof_info(dof_handler);
  MeshWorker::loop<2, 2, MeshWorker::DoFInfo<2>, MeshWorker::IntegrationInfoBox<2>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      discretization.info_box,
      cell_worker,
      NULL,
      NULL,
      assembler);

  return value;
}

} // namespace RadProblem
