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

  void renumber_to_quadrant(const unsigned int q);
  void setup();
  void setup_quadrant_renumbering();

  const AngularQuadrature & get_aq() const { return aq; }
  const DoFHandler<2> & get_dof_handler() const { return dof_handler; }
  const FE_DGQ<2> & get_fe() const { return fe; }
  const std::vector<unsigned int> & get_renumber_quadrant(const unsigned int q) const
  {
    return renumber_quadrant[q];
  }
  const std::vector<unsigned int> & get_renumber_ref_quadrant(const unsigned int q) const
  {
    return renumber_ref_quadrant[q];
  }
  unsigned int get_renumber_ref_quadrant(const unsigned int q, const unsigned int i)
  {
    return renumber_ref_quadrant[q][i];
  }
  const SparsityPattern & get_sparsity_quadrant(const unsigned int q) const
  {
    return sparsity_quadrant[q];
  }
  const SparsityPattern & get_sparsity_pattern() const { return sparsity_pattern; }
  bool & renumber_quadrants() { return renumber; }

  MeshWorker::IntegrationInfoBox<2> info_box;

private:
  void compare(std::vector<unsigned int> & num1, std::vector<unsigned int> & num2);

  Triangulation<2> triangulation;
  const MappingQ1<2> mapping;
  FE_DGQ<2> fe;
  DoFHandler<2> dof_handler;
  SparsityPattern sparsity_pattern;

  // Renumberings and patterns with downstream renumbering for each quadrant
  std::vector<std::vector<unsigned int>> renumber_quadrant;
  std::vector<std::vector<unsigned int>> renumber_ref_quadrant;
  std::vector<SparsityPattern> sparsity_quadrant;

  // The angular quadrature object
  AngularQuadrature aq;

  // Angular quadrature order
  unsigned int aq_order;
  // Mesh uniform refinements
  unsigned int uniform_refinement;
  // Whether or not to renumber quadrants
  bool renumber;
};
} // namespace SNProblem

#endif // DISCRETIZATION_H
