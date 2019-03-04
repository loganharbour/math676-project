#ifndef PROBLEM_H
#define PROBLEM_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

#include <map>

namespace SNProblem
{
using namespace dealii;

// Forward declarations
class AngularQuadrature;
class Description;
class Discretization;
class Material;

class Problem
{
public:
  Problem();

  void run();
  void output();

private:
  void setup();

  void assemble_direction(const Tensor<1, 2> dir);
  void solve_direction();
  void solve();

  Description description;
  Discretization discretization;

  const DoFHandler<2> & dof_handler;
  const std::map<const unsigned int, const Material> & materials;

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

  using DoFInfo = MeshWorker::DoFInfo<2>;
  using CellInfo = MeshWorker::IntegrationInfo<2>;
  void integrate_cell(DoFInfo & dinfo,
                      CellInfo & info,
                      const Tensor<1, 2> dir,
                      const unsigned int quadrant);
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Tensor<1, 2> dir);
  void integrate_face(DoFInfo & dinfo1,
                      DoFInfo & dinfo2,
                      CellInfo & info1,
                      CellInfo & info2,
                      const Tensor<1, 2> dir);
};
} // namespace SNProblem

#endif // PROBLEM_H
