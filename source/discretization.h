#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "angular_quadrature.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/meshworker/integration_info.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
class Discretization : public ParameterAcceptor
{
public:
  Discretization();

  void renumber_dofs(const unsigned int h);
  void setup();

  const AngularQuadrature<dim> & get_aq() const { return aq; }
  const std::set<unsigned int> & get_boundary_ids() const { return boundary_ids; }
  const DoFHandler<dim> & get_dof_handler() const { return dof_handler; }
  const MappingQ1<dim> & get_mapping() const { return mapping; }
  const std::set<unsigned int> & get_material_ids() const { return material_ids; }
  const std::vector<unsigned int> & get_ref_renumbering() const { return renumberings[0]; }
  const unsigned int & get_ref_renumbering(unsigned int i) const { return renumberings[0][i]; }
  const SparsityPattern & get_sparsity_pattern() const { return sparsity_pattern; }
  const bool & do_renumber() { return renumber; }

private:
  void generate_mesh();

  Triangulation<dim> triangulation;
  const MappingQ1<dim> mapping;
  FE_DGQ<dim> fe;
  DoFHandler<dim> dof_handler;
  SparsityPattern sparsity_pattern;

  /// The angular quadrature object
  AngularQuadrature<dim> aq;

  /// Renumberings between half ranges (if enabled)
  std::vector<std::vector<unsigned int>> renumberings;

  /// Boundary ids that exist on the mesh
  std::set<unsigned int> boundary_ids;
  /// Material ids that exist on the mesh
  std::set<unsigned int> material_ids;

  /// Angular quadrature order
  unsigned int aq_order = 10;
  /// Mesh uniform refinements
  unsigned int uniform_refinement = 0;
  /// Whether or not to renumber
  bool renumber = true;
  /// Hyper cube mesh bounds
  std::vector<double> hypercube_bounds = {0, 10};
};

} // namespace RadProblem

#endif // DISCRETIZATION_H
