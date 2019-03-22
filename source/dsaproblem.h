#ifndef DSA_PROBLEM
#define DSA_PROBLEM

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

namespace RadProblem
{

// Forward declarations
class AngularQuadrature;
class Description;
class Discretization;
class Problem;

using namespace dealii;
using DoFInfo = MeshWorker::DoFInfo<2>;
using CellInfo = MeshWorker::IntegrationInfo<2>;

class DSAProblem : public ParameterAcceptor
{
public:
  DSAProblem(Problem & problem);

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
  void integrate_cell_initial(DoFInfo & dinfo, CellInfo & info) const;
  /// Initial boundary integration term for MeshWorker
  void integrate_boundary_initial(DoFInfo & dinfo, CellInfo & info) const;
  /// Initial face integration term for MeshWorker
  void integrate_face_initial(DoFInfo & dinfo1,
                              DoFInfo & dinfo2,
                              CellInfo & info1,
                              CellInfo & info2) const;

  /// Assemble the components of the LHS and RHS that change with each iteration
  void assemble();
  /// Cell integration term for MeshWorker
  void integrate_cell(DoFInfo & dinfo, CellInfo & info) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info) const;

  /// Access to the description in the Problem
  const Description & description;
  /// Access to the discretization in the Problem
  const Discretization & discretization;
  /// Access to the dof_handler in the Description
  const DoFHandler<2> & dof_handler;
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
  MeshWorker::IntegrationInfoBox<2> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  /// Whether or not DSA is enabled
  bool enabled = true;
};
} // namespace RadProblem

#endif // DSA_PROBLEM
