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
  Problem(const Description & description, Discretization & discretization);

  void run();

private:
  void setup();

  void assemble_direction(const Point<2> dir);
  void solve_direction();
  void solve();

  const Description & description;
  Discretization & discretization;
  const DoFHandler<2> & dof_handler;
  const std::map<const unsigned int, const Material> & materials;

  SparseMatrix<double> system_matrix;
  Vector<double> rhs;
  Vector<double> phi_old;
  Vector<double> phi;
  Vector<double> solution;

  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  using DoFInfo = MeshWorker::DoFInfo<2>;
  using CellInfo = MeshWorker::IntegrationInfo<2>;
  void integrate_cell(DoFInfo & dinfo, CellInfo & info, const Point<2> dir);
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info, const Point<2> dir);
  void integrate_face(
      DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2, const Point<2> dir);
};
} // namespace SNProblem

#endif // PROBLEM_H
