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

  void setup();

  const AngularQuadrature & get_aq() const { return aq; }
  const DoFHandler<2> & get_dof_handler() const { return dof_handler; }
  const FE_DGQ<2> & get_fe() const { return fe; }
  const SparsityPattern & get_sparsity_pattern() const { return sparsity_pattern; }

  MeshWorker::IntegrationInfoBox<2> info_box;

private:
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
};
} // namespace SNProblem

#endif // DISCRETIZATION_H
