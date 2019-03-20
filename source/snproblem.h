#ifndef SNPROBLEM_H
#define SNPROBLEM_H

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

class SNProblem : public ParameterAcceptor
{
public:
  SNProblem(Problem & problem);

  /// Initial setup for the SNProblem
  void setup();

  /// Solve all directions
  bool solve_directions(const unsigned int l);

  const std::vector<double> & get_residuals() const { return residuals; }

private:

  /// Solve and fill scalar flux for angular direction d
  void solve_direction(const unsigned int d);

  /// Assemble LHS and RHS for angular direction d
  void assemble_direction(const Tensor<1, 2> & dir, const bool renumber_flux);
  /// Cell integration term for MeshWorker
  void integrate_cell(DoFInfo & dinfo,
                      CellInfo & info,
                      const Tensor<1, 2> dir,
                      const bool renumber_flux);
  /// Boundary integration term for MeshWorker
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Tensor<1, 2> dir);
  /// Face integration term for MeshWorker
  void integrate_face(DoFInfo & dinfo1,
                      DoFInfo & dinfo2,
                      CellInfo & info1,
                      CellInfo & info2,
                      const Tensor<1, 2> dir);

  /// Compute the L2 norm of (v1 - v2)
  double L2_difference(const Vector<double> & v1, const Vector<double> & v2);

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
  /// System solution used in solving a single direction
  Vector<double> solution;
  /// InfoBox for MeshWorker
  MeshWorker::IntegrationInfoBox<2> info_box;
  /// Assembler used by the MeshWorker::loop
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  /// Source iteration residuals
  std::vector<double> residuals;

  /// Source iteration tolerance
  double source_iteration_tolerance = 1.0e-12;
};
} // namespace RadProblem

#endif // SNPROBLEM_H
