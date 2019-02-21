#include "discretization.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/simple.h>

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

  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  const unsigned int n_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_points, n_points, n_points);
  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);
  info_box.initialize(fe, mapping);

  aq.init(aq_order);
}
} // namespace SNProblem
