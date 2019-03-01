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

  std::vector<unsigned int> temp(dof_handler.n_dofs());
  downstream_renumberings.resize(4, std::vector<unsigned int>(dof_handler.n_dofs()));
  sparsity_patterns.resize(4);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());

  DoFRenumbering::downstream(dof_handler, aq.dir(3 * aq.n_dir() / 4));

  for (unsigned int i = 0; i < 4; ++i)
  {
    unsigned int d = i * aq.n_dir() / 4;
    DoFRenumbering::compute_downstream(
        downstream_renumberings[i], temp, dof_handler, aq.dir(d), false);
    dof_handler.renumber_dofs(downstream_renumberings[i]);
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_patterns[i].copy_from(dsp);
  }
}

void
Discretization::renumber_dofs(const unsigned int i) {
  dof_handler.renumber_dofs(downstream_renumberings[i]);
}
} // namespace SNProblem
