#include "discretization.h"

#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/sparsity_tools.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
Discretization<dim>::Discretization(MPI_Comm & comm, TimerOutput & timer)
  : ParameterAcceptor("Discretization"),
    comm(comm),
    timer(timer),
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
  // Generate a mesh from gmsh
  add_parameter("msh", msh);

  // Split top and bottom material
  add_parameter("split_top_bottom", split_top_bottom);
}

template <int dim>
void
Discretization<dim>::setup()
{
  TimerOutput::Scope t(timer, "Discretization setup");

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
  if (hypercube_bounds.size() != 0 && msh.length() != 0)
    throw ExcMessage("msh and hypercube_bounds cannot be supplied together");
  if (hypercube_bounds.size() == 0 && split_top_bottom)
    throw ExcMessage("split_top_bottom only works with hypercube mesh");
  if (split_top_bottom && dim != 2)
    throw ExcMessage("split_top_bottom only works in 2D");
  if (hypercube_bounds.size() != 0 && hypercube_bounds.size() != 2)
    throw ExcMessage("hypercube_bounds must be of size 2 (lower and upper bounds)");

  // Generate hyper cube
  if (hypercube_bounds.size() != 0)
    GridGenerator::hyper_cube(triangulation, hypercube_bounds[0], hypercube_bounds[1], true);
  // Generate from gmsh
  else if (msh.length() != 0)
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(msh);
    gridin.read_msh(f);
  }

  // Refine if requested
  triangulation.refine_global(uniform_refinement);

  // Fill the boundary and material ids that exist on the local mesh
  for (auto & cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    // Set material in the upper half to 1 with split_top_bottom
    if (split_top_bottom && cell->center()[1] > (hypercube_bounds[0] + hypercube_bounds[1]) / 2)
      cell->set_material_id(1);

    local_material_ids.insert(cell->material_id());
    if (!cell->at_boundary())
      continue;
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      if (cell->face(i)->at_boundary())
        local_boundary_ids.insert(cell->face(i)->boundary_id());
  }

  // TODO: Communicate the boundary and material ids to all processors. Therefore,
  // this is hardcoded for now.
  if (hypercube_bounds.size() != 0)
  {
    if (dim == 2)
      boundary_ids = {0, 1, 2, 3};
    else
      boundary_ids = {0, 1, 2, 3, 4, 5};
  }
  else
    boundary_ids = {0};
  if (split_top_bottom || hypercube_bounds.size() == 0)
    material_ids = {0, 1};
  else
    material_ids = {0};
}

template Discretization<2>::Discretization(MPI_Comm & comm, TimerOutput & timer);
template Discretization<3>::Discretization(MPI_Comm & comm, TimerOutput & timer);

template void Discretization<2>::setup();
template void Discretization<3>::setup();
} // namespace RadProblem
