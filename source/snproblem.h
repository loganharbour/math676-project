#ifndef SNPROBLEM_H
#define SNPROBLEM_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

namespace RadProblem
{

using namespace dealii;

// Forward declarations
template <int dim>
class AngularQuadrature;
template <int dim>
class Description;
template <int dim>
class Discretization;
template <int dim>
class Problem;

template <int dim>
class SNProblem : public ParameterAcceptor
{
public:
  SNProblem(Problem<dim> & problem);

  /// Initial setup for the SNProblem
  void setup();

  /// Solve all directions
  void solve_directions();

private:
  /// Solve and fill scalar flux for angular direction d
  void solve_direction(const unsigned int d);

  /// Assemble LHS and RHS for angular direction d
  void assemble_direction(const Tensor<1, dim> & dir);
  /// Cell integration term for MeshWorker
  void integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                      MeshWorker::IntegrationInfo<dim> & info,
                      const Tensor<1, dim> & dir) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                          MeshWorker::IntegrationInfo<dim> & info,
                          const Tensor<1, dim> & dir) const;
  /// Face integration term for MeshWorker
  void integrate_face(MeshWorker::DoFInfo<dim> & dinfo1,
                      MeshWorker::DoFInfo<dim> & dinfo2,
                      MeshWorker::IntegrationInfo<dim> & info1,
                      MeshWorker::IntegrationInfo<dim> & info2,
                      const Tensor<1, dim> & dir) const;

  /// Access to the description in the Problem
  const Description<dim> & description;
  /// Access to the discretization in the Problem
  Discretization<dim> & discretization;
  /// Access to the dof_handler in the Description
  const DoFHandler<dim> & dof_handler;
  /// Access to the angular quadrature
  const AngularQuadrature<dim> & aq;
  /// Access the scalar flux DGFEM solution in the Problem
  Vector<double> & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  Vector<double> & scalar_flux_old;

  /// System matrix used in solving a single direction
  SparseMatrix<double> system_matrix;
  /// System right hand side used in solving a single direction
  Vector<double> system_rhs;
  /// Storage for the angular flux solution for a single direction
  Vector<double> solution;
  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<dim> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;
};
} // namespace RadProblem

#endif // SNPROBLEM_H
