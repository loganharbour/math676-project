#ifndef SNPROBLEM_H
#define SNPROBLEM_H

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>

#include "AngularQuadrature.h"
#include "Material.h"

using namespace dealii;

DeclException1(MaterialExists, int, "Material with id " << arg1 << " already exists.");

class SNProblem
{
public:
  SNProblem();
  void run();

  void add_material(const unsigned int id, const Material & material);

private:
  void setup_system();
  void assemble_direction(unsigned int d);
  void solve_direction(unsigned int d);
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  const MappingQ1<2> mapping;

  FE_DGQ<2> fe;
  DoFHandler<2> dof_handler;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> rhs;
  Vector<double> solution;

  // Scalar flux solution
  Vector<double> phi;

  // Material properties
  std::map<const unsigned int, const Material> materials;

  // Angular quadrature
  AngularQuadrature aq;

  MeshWorker::IntegrationInfoBox<2> info_box;
  MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;

  using DoFInfo = MeshWorker::DoFInfo<2>;
  using CellInfo = MeshWorker::IntegrationInfo<2>;

  void integrate_cell(DoFInfo & dinfo, CellInfo & info, Point<2> dir);
  void integrate_boundary(DoFInfo & dinfo, CellInfo & info, Point<2> dir);
  void integrate_face(
      DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2, Point<2> dir);
};

#endif // SNPROBLEM_H
