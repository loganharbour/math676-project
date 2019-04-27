#ifndef SNPROBLEM_H
#define SNPROBLEM_H

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
class SNProblem : public ParameterAcceptor
{
public:
  SNProblem(Problem<dim> & problem);

  /// Initial setup for the SNProblem
  void setup();

  /// Assemble, solve, and update all directions
  void assemble_solve_update();

  double get_reflective_DJ_norm() const { return reflective_dJ_norm; }

private:
  /// Solve and fill scalar flux for angular direction d
  void assemble_solve_update(const unsigned int d);
  /// Internal solve of the system for angular direction d
  void solve(const unsigned int d);

  /// Assemble LHS and RHS for angular direction d
  void assemble(const unsigned int d);
  /// Cell integration term for MeshWorker
  void integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                      MeshWorker::IntegrationInfo<dim> & info,
                      const unsigned int d) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                          MeshWorker::IntegrationInfo<dim> & info,
                          const unsigned int d) const;
  /// Face integration term for MeshWorker
  void integrate_face(MeshWorker::DoFInfo<dim> & dinfo1,
                      MeshWorker::DoFInfo<dim> & dinfo2,
                      MeshWorker::IntegrationInfo<dim> & info1,
                      MeshWorker::IntegrationInfo<dim> & info2,
                      const unsigned int d) const;

  /// Updates for reflective bcs before and after a single angular sweep
  void update_for_reflective_bc(const unsigned int d, const bool before_sweep);

  /// MPI communicator
  MPI_Comm & comm;
  /// Parallel cout
  ConditionalOStream pcout;
  /// Timer
  TimerOutput & timer;

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
  /// Access to the angular flux DGFEM solutions in the Problem
  std::vector<LA::MPI::Vector> & angular_flux;

  /// The unit normal for each reflective boundary
  std::map<types::global_dof_index, HatDirection> & reflective_dof_normals;
  /// Incoming angular flux on the reflective boundaries
  std::vector<std::map<types::global_dof_index, double>> & reflective_incoming_flux;
  /// Net current on the reflective boundaries
  std::map<types::global_dof_index, double> & reflective_dJ;

  /// System storage owned by the Problem
  LA::MPI::SparseMatrix & system_matrix;
  LA::MPI::Vector & system_rhs;
  LA::MPI::Vector & system_solution;

  /// Most recent norm of the net current on the reflecting boundary
  double reflective_dJ_norm = std::numeric_limits<double>::min();

  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<dim> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector> assembler;

  /// Whether or not to enable detailed solver output
  bool detailed_solver_output = false;
  /// Relative tolerance
  double relative_tolerance = 1e-12;
  /// Absolute tolerance
  double absolute_tolerance = 1e-12;
};
} // namespace RadProblem

#endif // SNPROBLEM_H
