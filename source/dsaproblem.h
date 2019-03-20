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
class Material;
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

  void solve();

  bool is_enabled() const { return enabled; }

private:
  void assemble();
  /// Cell integration term for MeshWorker
  void integrate_cell(DoFInfo & dinfo, CellInfo & info) const;
  /// Boundary integration term for MeshWorker
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info) const;
  /// Face integration term for MeshWorker
  void integrate_face(DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) const;

  /// Access to the description in the Problem
  const Description & description;
  /// Access to the discretization in the Problem
  Discretization & discretization;
  /// Access to the dof_handler in the Description
  const DoFHandler<2> & dof_handler;
  /// Access to the materials in Description
  const std::map<const unsigned int, const Material> & materials;
  /// Access to the angular quadrature
  const AngularQuadrature & aq;
  /// Access the scalar flux DGFEM solution in the Problem
  Vector<double> & scalar_flux;
  /// Access the old scalar flux DGFEM solution in the Problem
  Vector<double> & scalar_flux_old;

  /// System matrix used in solving a single direction
  SparseMatrix<double> matrix;
  /// System right hand side used in solving a single direction
  Vector<double> rhs;
  /// System scattering source/solution vector (used for both)
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