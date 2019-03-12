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

namespace SNProblem
{
using namespace dealii;

class Discretization : public ParameterAcceptor
{
public:
  Discretization();

  void get_material_ids(std::set<unsigned int> & material_ids) const;
  void renumber_dofs(const unsigned int h);
  void setup();
  void setup_renumbering();

  const AngularQuadrature & get_aq() const { return aq; }
  const DoFHandler<2> & get_dof_handler() const { return dof_handler; }
  const FE_DGQ<2> & get_fe() const { return fe; }
  const std::vector<unsigned int> & get_ref_renumbering() const { return renumberings[0]; }
  const unsigned int & get_ref_renumbering(unsigned int i) const { return renumberings[0][i]; }
  const SparsityPattern & get_sparsity_pattern() const { return sparsity_pattern; }
  const bool & do_renumber() { return renumber; }

  MeshWorker::IntegrationInfoBox<2> info_box;

private:
  void compare(std::vector<unsigned int> & num1, std::vector<unsigned int> & num2);

  Triangulation<2> triangulation;
  const MappingQ1<2> mapping;
  FE_DGQ<2> fe;
  DoFHandler<2> dof_handler;
  SparsityPattern sparsity_pattern;

  // The angular quadrature object
  AngularQuadrature aq;

  // Angular quadrature order
  unsigned int aq_order;
  // Mesh uniform refinements
  unsigned int uniform_refinement;

  // Whether or not to renumber
  bool renumber;
  // Renumberings between half ranges (if enabled)
  std::vector<std::vector<unsigned int>> renumberings;

};
} // namespace SNProblem

#endif // DISCRETIZATION_H
