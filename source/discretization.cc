#include "discretization.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/sparsity_tools.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
Discretization<dim>::Discretization(MPI_Comm & comm)
  : ParameterAcceptor("Discretization"),
    comm(comm),
    triangulation(comm),
    mapping(),
    fe(1),
    dof_handler(triangulation)
{
  // Angular quadrature order (default: 10)
  add_parameter("aq_order", aq_order);

  // Mesh uniform refinement levels (default: 0)
  add_parameter("uniform_refinement", uniform_refinement);

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

  // Distribute local degrees of freedom
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // Initialize angular quadrature
  aq.init(aq_order);

  // Setup sparsity pattern
  dsp.reinit(locally_relevant_dofs.size(), locally_relevant_dofs.size(), locally_relevant_dofs);
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(
      dsp, dof_handler.n_locally_owned_dofs_per_processor(), comm, locally_relevant_dofs);
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
  // TODO: Do this check correctly in parallel
  for (const auto & cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    material_ids.insert(cell->material_id());
    if (!cell->at_boundary())
      continue;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      if (cell->face(i)->at_boundary())
        boundary_ids.insert(cell->face(i)->boundary_id());
  }
}

template Discretization<1>::Discretization(MPI_Comm & comm);
template Discretization<2>::Discretization(MPI_Comm & comm);
template Discretization<3>::Discretization(MPI_Comm & comm);

template void Discretization<1>::setup();
template void Discretization<2>::setup();
template void Discretization<3>::setup();
} // namespace RadProblem
