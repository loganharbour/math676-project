#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>

namespace SNProblem
{
using namespace dealii;

Problem::Problem(const Description & description, Discretization & discretization)
  : description(description),
    discretization(discretization),
    dof_handler(discretization.get_dof_handler()),
    aq(discretization.get_aq()),
    materials(description.get_materials())
{
}

void
Problem::setup()
{
  phi.reinit(dof_handler.n_dofs());
  phi_old.reinit(dof_handler.n_dofs());
  rhs.reinit(dof_handler.n_dofs());
  solution.reinit(dof_handler.n_dofs());
  system_matrix.reinit(discretization.get_sparsity_pattern());

  assembler.initialize(system_matrix, rhs);
}

void
Problem::assemble_direction(unsigned int d)
{
  const auto dir = aq.dir(d);
  system_matrix = 0;
  rhs = 0;

  MeshWorker::DoFInfo<2> dof_info(dof_handler);

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    Problem::integrate_cell(dinfo, info, dir);
  };
  const auto boundary_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    Problem::integrate_boundary(dinfo, info, dir);
  };
  const auto face_worker =
      [&](DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) {
        Problem::integrate_face(dinfo1, dinfo2, info1, info2, dir);
      };

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
Problem::integrate_cell(DoFInfo & dinfo, CellInfo & info, Point<2> dir)
{
  const FEValuesBase<2> & fe_v = info.fe_values();
  const unsigned int n_q = fe_v.n_quadrature_points;
  const unsigned int n_dofs = fe_v.dofs_per_cell;

  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Get material for this cell
  const unsigned int material_id = dinfo.cell->material_id();
  Assert(materials.find(material_id) != materials.end(),
         ExcMessage("Material id not found in material map"));
  const Material & material = materials.at(material_id);
  const bool has_scattering = material.sigma_s != 0;

  // Obtain the old scalar flux if scattering exists in this cell
  std::vector<double> local_phi_old;
  if (has_scattering)
  {
    local_phi_old.resize(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      local_phi_old[i] = phi_old[dinfo.indices[i]];
  }

  for (unsigned int q = 0; q < n_q; ++q)
  {
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const double u_i = fe_v.shape_value(i, q);
      double phi_old_q = 0;
      for (unsigned int j = 0; j < n_dofs; ++j)
      {
        const double v_j = fe_v.shape_value(j, q);
        // Streaming term
        local_matrix(i, j) -= v_j * dir * fe_v.shape_grad(i, q) * JxW[q];
        // Loss term
        local_matrix(i, j) += u_i * v_j * material.sigma_t * JxW[q];
        // Accumulate scalar flux at this qp
        if (has_scattering)
          phi_old_q += local_phi_old[j] * v_j;
      }
      // Source gain term
      local_vector(i) += u_i * material.src * JxW[q];
      // Scattering gain term
      if (has_scattering)
        local_vector(i) += u_i * material.sigma_s * phi_old_q * JxW[q];
    }
  }
}

void
Problem::integrate_boundary(DoFInfo & /*dinfo*/, CellInfo & /*info*/, Point<2> /*dir*/)
{
}

void
Problem::integrate_face(
    DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2, Point<2> dir)
{
  const FEValuesBase<2> & fe1 = info1.fe_values();
  const FEValuesBase<2> & fe2 = info2.fe_values();
  const std::vector<double> & JxW = fe1.get_JxW_values();

  // Dot product between the direction and the outgoing normal of face 1
  const double dot = dir * fe1.normal_vector(0);

  // Cell 1 is outgoing
  if (dot > 0)
  {
    FullMatrix<double> & u1_v1_matrix = dinfo1.matrix(0, false).matrix;
    FullMatrix<double> & u1_v2_matrix = dinfo2.matrix(0, true).matrix;
    for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe1.dofs_per_cell; ++i)
      {
        const double u1_i = fe1.shape_value(i, q);
        for (unsigned int j = 0; j < fe1.dofs_per_cell; ++j)
          u1_v1_matrix(i, j) += dot * u1_i * fe1.shape_value(j, q) * JxW[q];
        for (unsigned int l = 0; l < fe2.dofs_per_cell; ++l)
          u1_v2_matrix(i, l) -= dot * u1_i * fe2.shape_value(l, q) * JxW[q];
      }
  }
  // Cell 2 is outgoing
  else if (dot < 0)
  {
    FullMatrix<double> & u2_v1_matrix = dinfo1.matrix(0, true).matrix;
    FullMatrix<double> & u2_v2_matrix = dinfo2.matrix(0, false).matrix;
    for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
      for (unsigned int k = 0; k < fe2.dofs_per_cell; ++k)
      {
        const double u2_k = fe2.shape_value(k, q);
        for (unsigned int j = 0; j < fe1.dofs_per_cell; ++j)
          u2_v1_matrix(k, j) += dot * u2_k * fe1.shape_value(j, q) * JxW[q];
        for (unsigned int l = 0; l < fe2.dofs_per_cell; ++l)
          u2_v2_matrix(k, l) -= dot * u2_k * fe2.shape_value(l, q) * JxW[q];
      }
  }
}

void
Problem::solve_direction()
{
  SolverControl solver_control(10000, 1e-10);
  SolverRichardson<> solver(solver_control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, discretization.get_fe().dofs_per_cell);
  solver.solve(system_matrix, solution, rhs, preconditioner);
}

void
Problem::solve()
{
  // Zero scalar flux
  phi = 0;

  for (unsigned int d = 0; d < aq.n_dir(); ++d)
  {
    std::cout << "Solving direction " << d << std::endl;

    // Assemble and solve
    assemble_direction(d);
    solve_direction();

    // Update scalar flux at each node
    phi.add(aq.w(d), solution);
  }

  phi_old = phi;
}

void
Problem::run()
{
  discretization.setup();

  setup();
  solve();
}
} // namespace SNProblem
