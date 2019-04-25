#ifndef DSA_PROBLEM
#define DSA_PROBLEM

#include "angular_quadrature.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

namespace LA = dealii::LinearAlgebraTrilinos;

namespace RadProblem
{

using namespace dealii;

// Forward declarations
template <int dim>
class Description;
template <int dim>
class Discretization;
template <int dim>
class Problem;

template <int dim>
class DSAProblem : public ParameterAcceptor
{
public:
  DSAProblem(Problem<dim> & problem);

  /// Initial setup for the DSAProblem
  void setup();
  /// Assemble, solve, and update DSA
  void assemble_solve_update();

private:
  /// Assemble the initial LHS, which remain constants
  void assemble_initial();
  /// Initial cell integration term for MeshWorker
  void integrate_cell_initial(MeshWorker::DoFInfo<dim> & dinfo,
                              MeshWorker::IntegrationInfo<dim> & info) const;
  /// Initial boundary integration term for MeshWorker
  void integrate_boundary_initial(MeshWorker::DoFInfo<dim> & dinfo,
                                  MeshWorker::IntegrationInfo<dim> & info) const;
  /// Initial face integration term for MeshWorker
  void integrate_face_initial(MeshWorker::DoFInfo<dim> & dinfo1,
                              MeshWorker::DoFInfo<dim> & dinfo2,
                              MeshWorker::IntegrationInfo<dim> & info1,
                              MeshWorker::IntegrationInfo<dim> & info2) const;

  /// Solve the system
  void solve();
  /// Assemble the components of the LHS and RHS that change with each iteration
  void assemble();
  /// Cell integration term for MeshWorker
  void integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                      MeshWorker::IntegrationInfo<dim> & info) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                          MeshWorker::IntegrationInfo<dim> & info) const;

  /// MPI communicator
  MPI_Comm & comm;
  /// Parallel cout
  ConditionalOStream pcout;
  /// Timer
  TimerOutput & timer;

  /// Access to the description in the Problem
  const Description<dim> & description;
  /// Access to the discretization in the Problem
  const Discretization<dim> & discretization;

  /// Access to the dof_handler in the Description
  const DoFHandler<dim> & dof_handler;
  /// Access to the angular quadrature
  const AngularQuadrature<dim> & aq;

  /// Access the scalar flux DGFEM solution in the Problem
  LA::MPI::Vector & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  const LA::MPI::Vector & scalar_flux_old;

  /// The unit normal for each reflective boundary
  const std::map<types::global_dof_index, HatDirection> & reflective_dof_normals;
  /// Incoming angular flux on the reflective boundaries
  std::vector<std::map<types::global_dof_index, double>> & reflective_incoming_flux;
  /// Net current on the reflective boundaries
  const std::map<types::global_dof_index, double> & reflective_dJ;

  /// System storage owned by the Problem
  LA::MPI::SparseMatrix & system_matrix;
  LA::MPI::Vector & system_rhs;
  LA::MPI::Vector & system_solution;

  /// System storage for the constant LHS
  LA::MPI::SparseMatrix dsa_matrix;

  /// Whether or not acceleration using dJ is enabled for reflective bcs
  bool reflective_bc_acceleration = true;
};
} // namespace RadProblem

#endif // DSA_PROBLEM
