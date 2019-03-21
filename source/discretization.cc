#include "discretization.h"

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/simple.h>

namespace RadProblem
{
using namespace dealii;

Discretization::Discretization()
  : ParameterAcceptor("Discretization"), mapping(), fe(1), dof_handler(triangulation)
{
  // Angular quadrature order (default: 10)
  add_parameter("aq_order", aq_order);

  // Mesh uniform refinement levels (default: 0)
  add_parameter("uniform_refinement", uniform_refinement);

  // Enable renumbering (default: true)
  add_parameter("renumber", renumber);

  // Generate a hyper cube mesh (default: {0, 10}); empty if no hypercube generation
  add_parameter("hypercube_bounds", hypercube_bounds);
}

void
Discretization::setup()
{
  // Generate the mesh depending on user inputs
  generate_mesh();

  // Distribute degrees of freedom
  dof_handler.distribute_dofs(fe);

  // Initialize angular quadrature
  aq.init(aq_order);

  // Default renumbering is downstream for the n_dir/2 direction
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFRenumbering::downstream(dof_handler, aq.dir(aq.n_dir() / 2), false);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  // Compute extra downstream numbering if requested
  if (renumber)
  {
    renumberings.resize(2, std::vector<unsigned int>(dof_handler.n_dofs()));
    DoFRenumbering::compute_downstream(
        renumberings[0], renumberings[1], dof_handler, aq.dir(0), false);
  }
}

void
Discretization::generate_mesh()
{
  if (hypercube_bounds.size() != 0 && hypercube_bounds.size() != 2)
    throw ExcMessage("hypercube_bounds must be of size 2 (lower and upper bounds)");
  if (hypercube_bounds.size() == 0)
    throw ExcMessage("hypercube_bounds must be set (no other mesh currently supported)");

  // Generate hyper cube
  GridGenerator::hyper_cube(triangulation, hypercube_bounds[0], hypercube_bounds[1]);

  // Refine if requested
  triangulation.refine_global(uniform_refinement);
}

void
Discretization::get_material_ids(std::set<unsigned int> & material_ids) const
{
  material_ids.clear();
  for (const auto & cell : dof_handler.active_cell_iterators())
    material_ids.insert(cell->material_id());
}

void
Discretization::renumber_dofs(const unsigned int half)
{
  if (half != 0 && half != 1)
    throw ExcMessage("half can only be 0 or 1 in renumber_dofs");
  dof_handler.renumber_dofs(renumberings[half]);
}

} // namespace RadProblem
