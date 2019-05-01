#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "angular_quadrature.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
class Discretization : public ParameterAcceptor
{
public:
  Discretization(MPI_Comm & comm, TimerOutput & timer);

  void setup();

  const AngularQuadrature<dim> & get_aq() const { return aq; }
  const std::set<types::boundary_id> & get_boundary_ids() const { return boundary_ids; }
  const DoFHandler<dim> & get_dof_handler() const { return dof_handler; }
  const DynamicSparsityPattern & get_sparsity_pattern() const { return dsp; }

  const MappingQ1<dim> & get_mapping() const { return mapping; }
  const std::set<types::material_id> & get_material_ids() const { return material_ids; }
  const IndexSet & get_locally_owned_dofs() const { return locally_owned_dofs; }
  const parallel::distributed::Triangulation<dim> & get_triangulation() const
  {
    return triangulation;
  }

private:
  /// Generate the mesh
  void generate_mesh();

  /// MPI communicator
  MPI_Comm & comm;
  /// Timer
  TimerOutput & timer;

  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ1<dim> mapping;
  FE_DGQ<dim> fe;
  DoFHandler<dim> dof_handler;
  DynamicSparsityPattern dsp;

  /// Local degrees of freedom
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  /// The angular quadrature object
  AngularQuadrature<dim> aq;

  /// Boundary ids that exist on the entire mesh
  std::set<types::boundary_id> boundary_ids;
  /// Material ids that exist on the entire mesh
  std::set<types::material_id> material_ids;

  /// Angular quadrature order
  unsigned int aq_order = 10;
  /// Angular quadrature type
  //AQ_Type aq_type = RadProblem::product;
  unsigned int aq_type = 0;
  /// Mesh uniform refinements
  unsigned int uniform_refinement = 0;
  /// Hyper cube mesh bounds
  std::vector<double> hypercube_bounds = {};
  /// Split top and bottom material
  bool split_top_bottom = false;
  /// Gmsh filename
  std::string msh;
};

} // namespace RadProblem

#endif // DISCRETIZATION_H
