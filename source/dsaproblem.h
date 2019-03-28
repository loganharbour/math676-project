#ifndef DSA_PROBLEM
#define DSA_PROBLEM

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

namespace LA = dealii::LinearAlgebraTrilinos::MPI;

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
  // Solve the DSAProblem
  void solve();

  /// Whether or not DSA is enabled
  bool is_enabled() const { return enabled; }

private:
  /// Assemble the initial LHS and RHS (dsa_matrix, dsa_rhs), which remain constant
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

  /// Assemble the components of the LHS and RHS that change with each iteration
  void assemble();
  /// Cell integration term for MeshWorker
  void integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                      MeshWorker::IntegrationInfo<dim> & info) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                          MeshWorker::IntegrationInfo<dim> & info) const;

  /// Access to the description in the Problem
  const Description<dim> & description;
  /// Access to the discretization in the Problem
  const Discretization<dim> & discretization;

  /// Access to the dof_handler in the Description
  const DoFHandler<dim> & dof_handler;

  /// Access the scalar flux DGFEM solution in the Problem
  LA::Vector & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  const LA::Vector & scalar_flux_old;

  /// System storage owned by the Problem
  LA::SparseMatrix & system_matrix;
  LA::Vector & system_rhs;
  LA::Vector & system_solution;

  /// System storage for the constant LHS and RHS
  LA::SparseMatrix dsa_matrix;
  LA::Vector dsa_rhs;

  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<dim> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<LA::MPI::SparseMatrix, LA::MPI::Vector> assembler;

  /// Whether or not DSA is enabled
  bool enabled = true;
};
} // namespace RadProblem

#endif // DSA_PROBLEM
