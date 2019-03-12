#ifndef PROBLEM_H
#define PROBLEM_H

#include "description.h"
#include "discretization.h"
#include "material.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

#include <map>

namespace SNProblem
{
using namespace dealii;

using DoFInfo = MeshWorker::DoFInfo<2>;
using CellInfo = MeshWorker::IntegrationInfo<2>;

// Forward declarations
class AngularQuadrature;

class Problem : public ParameterAcceptor
{
public:
  Problem();

  void run();

private:
  void setup();

  void solve();

  void solve_direction(const unsigned int d);
  void assemble_direction(const Tensor<1, 2> & dir, const bool renumber_flux);
  void integrate_cell(DoFInfo & dinfo,
                      CellInfo & info,
                      const Tensor<1, 2> dir,
                      const bool renumber_flux);
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Tensor<1, 2> dir);
  void integrate_face(DoFInfo & dinfo1,
                      DoFInfo & dinfo2,
                      CellInfo & info1,
                      CellInfo & info2,
                      const Tensor<1, 2> dir);
  void solve_richardson();
  void solve_gauss_seidel();
  void update_scalar_flux(const double weight, const bool renumber_flux);

  double L2_difference(const Vector<double> & v1, const Vector<double> & v2);
  void output() const;
  void output_vtu() const;

  Description description;
  Discretization discretization;

  const DoFHandler<2> & dof_handler;
  const std::map<const unsigned int, const Material> & materials;
  const AngularQuadrature & aq;

  /// System matrix used in solving a single direction
  SparseMatrix<double> direction_matrix;
  /// System right hand side used in solving a single direction
  Vector<double> direction_rhs;
  /// System solution used in solving a single direction
  Vector<double> direction_solution;

  /// Finite element representation of the scalar flux at the current iteration
  Vector<double> scalar_flux;
  /// Finite element representation of the scalar flux at the previous iteration
  Vector<double> scalar_flux_old;

  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  /// Vtu output filename
  std::string vtu_filename = "output";
  /// Source iteration tolerance
  double source_iteration_tolerance = 1.0e-12;
};
} // namespace SNProblem

#endif // PROBLEM_H
