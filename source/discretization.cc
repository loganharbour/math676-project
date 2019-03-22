#include "discretization.h"

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/meshworker/simple.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
Discretization<dim>::Discretization()
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

template <int dim>
void
Discretization<dim>::setup()
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

template <int dim>
void
Discretization<dim>::generate_mesh()
{
  if (hypercube_bounds.size() != 0 && hypercube_bounds.size() != 2)
    throw ExcMessage("hypercube_bounds must be of size 2 (lower and upper bounds)");
  if (hypercube_bounds.size() == 0)
    throw ExcMessage("hypercube_bounds must be set (no other mesh currently supported)");

  // Generate hyper cube
  GridGenerator::hyper_cube(triangulation, hypercube_bounds[0], hypercube_bounds[1]);

  // Refine if requested
  triangulation.refine_global(uniform_refinement);

  // Fill the boundary and material ids that exist on the mesh
  for (const auto & cell : dof_handler.active_cell_iterators())
  {
    material_ids.insert(cell->material_id());
    if (!cell->at_boundary())
      continue;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      if (cell->face(i)->at_boundary())
        boundary_ids.insert(cell->face(i)->boundary_id());
  }
}

template <int dim>
void
Discretization<dim>::renumber_dofs(const unsigned int half)
{
  if (half != 0 && half != 1)
    throw ExcMessage("half can only be 0 or 1 in renumber_dofs");
  dof_handler.renumber_dofs(renumberings[half]);
}

template Discretization<1>::Discretization();
template Discretization<2>::Discretization();
template Discretization<3>::Discretization();

template void Discretization<1>::setup();
template void Discretization<2>::setup();
template void Discretization<3>::setup();

template void Discretization<1>::renumber_dofs(const unsigned int half);
template void Discretization<2>::renumber_dofs(const unsigned int half);
template void Discretization<3>::renumber_dofs(const unsigned int half);

} // namespace RadProblem
