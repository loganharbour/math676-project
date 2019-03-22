#include "dsaproblem.h"

#include "description.h"
#include "discretization.h"
#include "problem.h"

#include <deal.II/algorithms/any_data.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>

namespace RadProblem
{
using namespace dealii;

DSAProblem::DSAProblem(Problem & problem)
  : ParameterAcceptor("DSAProblem"),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old())
{
  // Whether or not DSA is enabled (default: true)
  add_parameter("enabled", enabled);
}

void
DSAProblem::setup()
{
  // Do not setup without scattering or if it is disabled
  if (!description.has_scattering() || !enabled)
    return;

  // Initialize system storage for a single direction
  rhs.reinit(dof_handler.n_dofs());
  matrix.reinit(discretization.get_sparsity_pattern());
  solution.reinit(dof_handler.n_dofs());

  // Setup InfoBox for MeshWorker
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(discretization.get_fe(), discretization.get_mapping());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(matrix, rhs);
}

void
DSAProblem::solve()
{
  // Skip if disabled
  if (!enabled)
    return;

  // Assembly
  assemble();

  // Solve system
  SolverControl control(1000, 1.e-12);
  SolverCG<Vector<double>> solver(control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(matrix, discretization.get_fe().dofs_per_cell);
  solver.solve(matrix, solution, rhs, preconditioner);
  std::cout << "  DSA converged after " << control.last_step() << " CG iterations" << std::endl;

  // Update scalar flux with change
  scalar_flux += solution;
}

void
DSAProblem::assemble()
{
  // Zero lhs and rhs before assembly
  matrix = 0;
  rhs = 0;

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    DSAProblem::integrate_cell(dinfo, info);
  };
  const auto boundary_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    DSAProblem::integrate_boundary(dinfo, info);
  };
  const auto face_worker =
      [&](DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) {
        DSAProblem::integrate_face(dinfo1, dinfo2, info1, info2);
      };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<2> dof_info(dof_handler);
  MeshWorker::loop<2, 2, MeshWorker::DoFInfo<2>, MeshWorker::IntegrationInfoBox<2>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      cell_worker,
      boundary_worker,
      face_worker,
      assembler);
}

void
DSAProblem::integrate_cell(DoFInfo & dinfo, CellInfo & info) const
{
  const FEValuesBase<2> & fe = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const auto & material = description.get_material(dinfo.cell->material_id());

  // Change in scalar flux at each vertex for the scattering "source"
  std::vector<double> scalar_flux_change(fe.dofs_per_cell);
  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    scalar_flux_change[i] = scalar_flux(dinfo.indices[i]) - scalar_flux_old(dinfo.indices[i]);

  LocalIntegrators::L2::mass_matrix(local_matrix, fe, material.sigma_a);
  LocalIntegrators::Laplace::cell_matrix(local_matrix, fe, material.D);
  LocalIntegrators::L2::L2(local_vector, fe, scalar_flux_change, material.sigma_s);
}

void
DSAProblem::integrate_face(DoFInfo & dinfo1,
                           DoFInfo & dinfo2,
                           CellInfo & info1,
                           CellInfo & info2) const
{
  const auto & fe1 = info1.fe_values();
  const auto & fe2 = info2.fe_values();

  // Diffusion coefficient in each cell
  const double D1 = description.get_material(dinfo1.cell->material_id()).D;
  const double D2 = description.get_material(dinfo2.cell->material_id()).D;

  // Length of the cells in the orthogonal direction to this face
  const unsigned int n1 = GeometryInfo<2>::unit_normal_direction[dinfo1.face_number];
  const unsigned int n2 = GeometryInfo<2>::unit_normal_direction[dinfo2.face_number];
  const double h1 = dinfo1.cell->extent_in_direction(n1);
  const double h2 = dinfo2.cell->extent_in_direction(n2);
  // Polynomial degrees on each face
  const unsigned int deg1 = fe1.get_fe().tensor_degree();
  const unsigned int deg2 = fe2.get_fe().tensor_degree();
  // Penalty coefficient
  const double c1 = 2 * deg1 * (deg1 + 1);
  const double c2 = 2 * deg2 * (deg2 + 1);
  const double kappa = std::fmax(0.5 * (c1 * D1 / h1 + c2 * D2 / h2), 0.25);

  FullMatrix<double> & v1u1 = dinfo1.matrix(0, false).matrix;
  FullMatrix<double> & v1u2 = dinfo1.matrix(0, true).matrix;
  FullMatrix<double> & v2u1 = dinfo2.matrix(0, true).matrix;
  FullMatrix<double> & v2u2 = dinfo2.matrix(0, false).matrix;
  const unsigned int n_dofs = fe1.dofs_per_cell;
  for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
  {
    const double JxW = fe1.JxW(q);
    const Tensor<1, 2> n = fe1.normal_vector(q);
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
        v1u2(i, j) += JxW * (-kappa * v1 * u2 + 0.5 * (-v1 * Ddnu2 - Ddnv1 * u2));
        v2u1(i, j) += JxW * (-kappa * v2 * u1 + 0.5 * (+v2 * Ddnu1 + Ddnv2 * u1));
        v2u2(i, j) += JxW * (+kappa * v2 * u2 + 0.5 * (-v2 * Ddnu2 - Ddnv2 * u2));
      }
    }
  }
}

void
DSAProblem::integrate_boundary(DoFInfo & dinfo, CellInfo & info) const
{
  // Bounday condition type for this face
  const auto bc_type = description.get_bc(dinfo.face->boundary_id()).type;

  // Diffusion coefficient
  const double D = description.get_material(dinfo.cell->material_id()).D;

  // RHS contribution (for dirichlet BCs)
  if (bc_type != Description::BCTypes::Reflective)
  {
    // Length of the cell in the orthogonal direction to this face
    const unsigned int n = GeometryInfo<2>::unit_normal_direction[dinfo.face_number];
    const double h = dinfo.cell->extent_in_direction(n);
    // Polynomial degrees on the face
    const unsigned int deg = info.fe_values().get_fe().tensor_degree();
    // Penalty coefficient
    const double c = 2 * deg * (deg + 1);
    const double kappa = std::fmax(c * D / h, 0.25);

    // Multiply penalty coefficient by 2 / D to negate the factor D / 2 that
    // multiplies every term in nitsche_matrix
    LocalIntegrators::Laplace::nitsche_matrix(
        dinfo.matrix(0, false).matrix, info.fe_values(), 2 * kappa / D, D / 2);
  }
}

} // namespace RadProblem
