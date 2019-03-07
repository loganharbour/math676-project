#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace SNProblem
{
using namespace dealii;

Problem::Problem()
  : dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
    renumber(discretization.renumber_quadrants())
{
}

void
Problem::setup()
{
  // Initialize system storage for a single direction
  direction_rhs.reinit(dof_handler.n_dofs());
  direction_matrix.reinit(discretization.get_sparsity_pattern());

  // Resize system variables
  direction_solution.reinit(dof_handler.n_dofs());
  scalar_flux.reinit(dof_handler.n_dofs());
  scalar_flux_old.reinit(dof_handler.n_dofs());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(direction_matrix, direction_rhs);
}

void
Problem::assemble_direction(const unsigned int d)
{
  const AngularQuadrature & aq = discretization.get_aq();
  const Tensor<1, 2> dir = aq.dir(d);
  const unsigned int quadrant = aq.quadrant(d);

  // Zero lhs and rhs before assembly
  direction_matrix = 0;
  direction_rhs = 0;

  MeshWorker::DoFInfo<2> dof_info(dof_handler);

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    Problem::integrate_cell(dinfo, info, dir, quadrant);
  };
  const auto boundary_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    Problem::integrate_boundary(dinfo, info, dir);
  };
  const auto face_worker =
      [&](DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) {
        Problem::integrate_face(dinfo1, dinfo2, info1, info2, dir);
      };

  // Call loop to execute the integration
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
Problem::integrate_cell(DoFInfo & dinfo,
                        CellInfo & info,
                        const Tensor<1, 2> dir,
                        const unsigned int quadrant)
{
  const FEValuesBase<2> & fe_v = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Get material for this cell
  Assert(materials.find(dinfo.cell->material_id()) != materials.end(),
         ExcMessage("Material id not found in material map"));
  const Material & material = materials.at(dinfo.cell->material_id());

  // Whether or not this cell has scattering
  const bool has_scattering = material.sigma_s != 0;

  // Obtain the old scalar flux if scattering exists in this cell
  std::vector<double> local_scalar_flux_old;
  if (has_scattering)
  {
    unsigned int index;
    local_scalar_flux_old.resize(fe_v.dofs_per_cell);
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      if (renumber && quadrant != 3)
        index = discretization.get_renumber_ref_quadrant(quadrant, dinfo.indices[i]);
      else
        index = dinfo.indices[i];
      local_scalar_flux_old[i] = scalar_flux_old[index];
    }
  }

  for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      const double v_i = fe_v.shape_value(i, q);
      const double dir_dot_grad_v_i = dir * fe_v.shape_grad(i, q);
      // Sampled scalar flux used for scattering source at this qp
      double scalar_flux_old_q = 0;
      for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
      {
        const double u_j = fe_v.shape_value(j, q);
        // Streaming + collision
        local_matrix(i, j) += (u_j * JxW[q]) * (material.sigma_t * v_i - dir_dot_grad_v_i);
        // Accumulate old scalar flux
        if (has_scattering)
          scalar_flux_old_q += scalar_flux_old[j] * v_i * JxW[q];
      }
      // External source
      local_vector(i) += v_i * material.src * JxW[q];
      // Scattering gain term
      if (has_scattering)
        local_vector(i) += v_i * material.sigma_s * scalar_flux_old_q * JxW[q];
    }
}

void
Problem::integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Tensor<1, 2> dir)
{
  // Dot product between the direction and the outgoing normal
  const double dir_dot_n = dir * info.fe_values().normal_vector(0);

  // Vacuum boundary conditions for now: no incoming flux
  if (dir_dot_n < 0)
    return;

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

void
Problem::integrate_face(
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
Problem::solve_direction()
{
  if (!renumber)
  {
    SolverControl solver_control(1000, 1e-12);
    SolverRichardson<> solver(solver_control);
    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(direction_matrix, discretization.get_fe().dofs_per_cell);
    solver.solve(direction_matrix, direction_solution, direction_rhs, preconditioner);
    std::cout << "  Converged after " << solver_control.last_step() << " iterations " << std::endl;
  }
  else
  {
    PreconditionBlockSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(direction_matrix, discretization.get_fe().dofs_per_cell);
    preconditioner.step(direction_solution, direction_rhs);
  }
}

void
Problem::solve()
{
  // Zero scalar flux for update
  scalar_flux = 0;

  const AngularQuadrature & aq = discretization.get_aq();
  for (unsigned int d = 0; d < aq.n_dir(); ++d)
  {
    const unsigned int quadrant = aq.quadrant(d);
    const double weight = aq.w(d);

    // Renumber at the beginning of a quadrant
    if (renumber && (d == 0 || quadrant != aq.quadrant(d - 1)))
    {
      std::cout << "Renumbering dofs for quadrant " << quadrant << std::endl;
      discretization.renumber_to_quadrant(quadrant);
      direction_matrix.reinit(discretization.get_sparsity_quadrant(quadrant));
    }

    std::cout << "Solving direction " << d << std::endl;

    // Assemble and solve
    assemble_direction(d);
    solve_direction();

    // Update scalar flux at each node (weighed by angular weight)
    if (quadrant == 3 || !renumber)
      scalar_flux.add(weight, direction_solution);
    else
    {
      std::cout << "to ref" << std::endl;
      const auto & to_ref = discretization.get_renumber_ref_quadrant(quadrant);
      for (unsigned int i = 0; i < discretization.get_fe().dofs_per_cell; ++i)
        scalar_flux[to_ref[i]] += weight * direction_solution[i];
    }
  }

  const auto & test1 = discretization.get_renumber_ref_quadrant(0);
  const auto & test2 = discretization.get_renumber_quadrant(3);
  std::cout << test1[0] << std::endl;
  std::cout << test2[test1[0]] << std::endl;
  scalar_flux_old = scalar_flux;
}

void
Problem::run()
{
  // Call setup on each class
  description.setup();
  discretization.setup();
  setup();

  // And... solve
  solve();
}

void
Problem::output()
{
  std::ofstream output("solution.vtu");
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(scalar_flux, "scalar_flux");
  data_out.build_patches();
  data_out.write_vtu(output);
}
} // namespace SNProblem
