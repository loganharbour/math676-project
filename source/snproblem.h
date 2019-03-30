#ifndef SNPROBLEM_H
#define SNPROBLEM_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

namespace LA = dealii::LinearAlgebraTrilinos;

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

  /// MPI communicator
  MPI_Comm & comm;
  /// Parallel cout
  ConditionalOStream pcout;
  
  /// Access to the description in the Problem
  const Description<dim> & description;
  /// Access to the discretization in the Problem
  Discretization<dim> & discretization;

  /// Access to the dof_handler in the Description
  const DoFHandler<dim> & dof_handler;
  /// Access to the angular quadrature
  const AngularQuadrature<dim> & aq;

  /// Access the scalar flux DGFEM solution in the Problem
  LA::MPI::Vector & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  LA::MPI::Vector & scalar_flux_old;

  /// System storage owned by the Problem
  LA::MPI::SparseMatrix & system_matrix;
  LA::MPI::Vector & system_rhs;
  LA::MPI::Vector & system_solution;


  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<dim> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector> assembler;
};
} // namespace RadProblem

#endif // SNPROBLEM_H
