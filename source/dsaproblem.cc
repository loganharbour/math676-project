#include "dsaproblem.h"

#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>

namespace RadProblem
{
using namespace dealii;

DSAProblem::DSAProblem(Problem & problem)
  : ParameterAcceptor("DSAProblem"),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
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
DSAProblem::integrate_cell(DoFInfo & dinfo, CellInfo & info) const
{
  const Material & material = description.get_material(dinfo.cell->material_id());

  auto & matrix = dinfo.matrix(0, false);
  LocalIntegrators::L2::mass_matrix(matrix, info.fe_values(), material.sigma_a);
  LocalIntegrators::Laplace::cell_matrix(matrix, info.fe_values(), material.D);
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

  // Penalty coefficient
  const unsigned int d1 = info1.fe_values(0).get_fe().tensor_degree();
  const unsigned int d2 = info2.fe_values(0).get_fe().tensor_degree();
  const unsigned int n1 = GeometryInfo<2>::unit_normal_direction[dinfo1.face_number];
  const unsigned int n2 = GeometryInfo<2>::unit_normal_direction[dinfo2.face_number];
  const double c1 = 2 * ((d1 == 0) ? 1 : d1 * (d1 + 1));
  const double c2 = 2 * ((d2 == 0) ? 1 : d2 * (d2 + 1));
  const double penalty1 = c1 * D1 / dinfo1.cell->extent_in_direction(n1);
  const double penalty2 = c2 * D2 / dinfo1.cell->extent_in_direction(n2);
  const double kappa = std::fmax(0.5 * (penalty1 + penalty2), 0.25);
}

void
DSAProblem::integrate_boundary(DoFInfo & dinfo,
                           CellInfo & info) const
{
  // Diffusion coefficient
  const double D = description.get_material(dinfo.cell->material_id()).D;

  // Penalty coefficient
  const unsigned int d = info.fe_values(0).get_fe().tensor_degree();
  const unsigned int n = GeometryInfo<2>::unit_normal_direction[dinfo.face_number];
  const double c = 2 * ((d == 0) ? 1 : d * (d + 1));
  const double penalty = c * D / dinfo.cell->extent_in_direction(n);
  const double kappa = std::fmax(penalty, 0.25);
}

} // namespace RadProblem
