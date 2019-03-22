#ifndef DSA_PROBLEM
#define DSA_PROBLEM

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

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
  Vector<double> & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  const Vector<double> & scalar_flux_old;

  /// Matrix that holds the constant DSA LHS (filled once on setup)
  SparseMatrix<double> dsa_matrix;
  /// Matrix used for solving the problem
  SparseMatrix<double> system_matrix;
  /// Vector that holds the constant DSA RHS (filled once on setup)
  Vector<double> dsa_rhs;
  /// RHS vector used for solving the problem
  Vector<double> system_rhs;
  /// Storage for the error corrector solution
  Vector<double> solution;
  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<dim> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  /// Whether or not DSA is enabled
  bool enabled = true;
};
} // namespace RadProblem

#endif // DSA_PROBLEM
