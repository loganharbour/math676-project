#include "discretization.h"

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/simple.h>

#include <fstream>
#include <string>

namespace SNProblem
{
using namespace dealii;

Discretization::Discretization()
  : ParameterAcceptor("Discretization"), mapping(), fe(1), dof_handler(triangulation)
{
  aq_order = 10;
  add_parameter("aq_order", aq_order);

  uniform_refinement = 0;
  add_parameter("uniform_refinement", uniform_refinement);
}

void
Discretization::setup()
{
  GridGenerator::hyper_cube(triangulation, 0, 10);
  triangulation.refine_global(uniform_refinement);

  // Distribute degrees of freedom
  dof_handler.distribute_dofs(fe);

  // Setup InfoBox for MeshWorker
  const unsigned int n_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_points, n_points, n_points);
  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.initialize(fe, mapping);

  // Initialize angular quadrature
  aq.init(aq_order);

  // Default renumbering is downstream for the first direction in the last quadrant
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFRenumbering::downstream(dof_handler, aq.dir(3 * aq.n_dir() / 4));
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  setup_quadrant_renumbering();
}

void
Discretization::setup_quadrant_renumbering()
{
  // Sparsity pattern we will use multiple times
  DynamicSparsityPattern dsp(dof_handler.n_dofs());

  // Temprorary vector for the other input argument to compute_downstream
  std::vector<unsigned int> temp(dof_handler.n_dofs());

  // Renumberings from 3 -> 0, 0 -> 1, 1 -> 2, 2 -> 3
  renumber_quadrant.resize(4, std::vector<unsigned int>(dof_handler.n_dofs()));
  // Sparsity patterns for quadrants 0, 1, 2, 3
  sparsity_quadrant.resize(4);
  // Renumberings from 0 -> 3, 1 -> 3, 2 -> 3
  renumber_ref_quadrant.resize(3, std::vector<unsigned int>(dof_handler.n_dofs()));

  // Reference quadrant direction
  const auto ref_dir = aq.dir(3 * aq.n_dir() / 4);

  // Store renumberings for quadrant q from the quadrant before it and
  // renumberings for quadrant q to the reference quadrant. By default, we
  // are currently renumbered to the final quadrant; therefore, setting for
  // quadrant 0 does the correct renumbering from quadrant 3 -> 0.
  for (unsigned int q = 0; q < 4; ++q)
  {
    // Last direction for this quadrant
    const auto dir = aq.dir(q * aq.n_dir() / 4);

    // Renumbering from quadrant q - 1 to quadrant q
    DoFRenumbering::compute_downstream(
        renumber_quadrant[q], temp, dof_handler, dir, false);

    // Store sparsity pattern for quadrant q
    dof_handler.renumber_dofs(renumber_quadrant[q]);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_quadrant[q].copy_from(dsp);

    // Renumbering from quadrant q to the reference (quadrant 3)
    if (q != 3)
      DoFRenumbering::compute_downstream(
          renumber_ref_quadrant[q], temp, dof_handler, ref_dir, false);
  }
}
void
Discretization::renumber_to_quadrant(const unsigned int q) {
  dof_handler.renumber_dofs(renumber_quadrant[q]);
}
} // namespace SNProblem
