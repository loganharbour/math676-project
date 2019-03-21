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
  // Diffusion coefficient in each cell
  const double D1 = description.get_material(dinfo1.cell->material_id()).D;
  const double D2 = description.get_material(dinfo2.cell->material_id()).D;
  // Polynomial degrees on each face
  const unsigned int deg1 = info1.fe_values().get_fe().tensor_degree();
  const unsigned int deg2 = info2.fe_values().get_fe().tensor_degree();
  // Length of the cells in the orthogonal direction to this face
  const double h1 = dinfo1.face->measure() / dinfo1.cell->measure();
  const double h2 = dinfo2.face->measure() / dinfo2.cell->measure();
  // Penalty coefficient
  const double c1 = 2 * deg1 * (deg1 + 1);
  const double c2 = 2 * deg2 * (deg2 + 1);
  const double kappa = std::fmax(0.5 * (c1 * D1 / h1 + c2 * D2 / h2), 0.25);

  // Multiply penalty coefficient by 2 / (D1 + D2) to cancel out
  // nu = (nui + nue) / 2 = 2 / (D1 + D2) in ip_matrix
  LocalIntegrators::Laplace::ip_matrix(dinfo1.matrix(0, false).matrix,
                                       dinfo1.matrix(0, true).matrix,
                                       dinfo2.matrix(0, true).matrix,
                                       dinfo2.matrix(0, false).matrix,
                                       info1.fe_values(),
                                       info2.fe_values(),
                                       2 * kappa / (D1 + D2),
                                       D1,
                                       D2);
}

void
DSAProblem::integrate_boundary(DoFInfo & dinfo, CellInfo & info) const
{
  const auto bc_type = description.get_bc(dinfo.face->boundary_id()).type;

  // Diffusion coefficient
  const double D = description.get_material(dinfo.cell->material_id()).D;

  // RHS contribution (for dirichlet BCs)
  if (bc_type != Description::BCTypes::Reflective)
  {
    // Polynomial degrees on the face
    const unsigned int deg = info.fe_values().get_fe().tensor_degree();
    // Length of the cell in the orthogonal direction to this face
    const double h = dinfo.face->measure() / dinfo.cell->measure();
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
