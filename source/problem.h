#ifndef PROBLEM_H
#define PROBLEM_H

#include "description.h"
#include "discretization.h"
#include "dsaproblem.h"
#include "snproblem.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA = dealii::LinearAlgebraTrilinos;

namespace RadProblem
{
using namespace dealii;

template <int dim>
class Problem : public ParameterAcceptor
{
public:
  Problem();

  void run();

  MPI_Comm & get_comm() { return comm; }
  ConditionalOStream & get_pcout() { return pcout; }
  TimerOutput & get_timer() { return timer; }

  const Description<dim> & get_description() const { return description; }
  const Discretization<dim> & get_discretization() const { return discretization; }

  LA::MPI::Vector & get_scalar_flux() { return scalar_flux; }
  LA::MPI::Vector & get_scalar_flux_old() { return scalar_flux_old; }
  const LA::MPI::Vector & get_scalar_flux_old() const { return scalar_flux_old; }
  std::vector<LA::MPI::Vector> & get_angular_flux() { return angular_flux; }

  LA::MPI::SparseMatrix & get_system_matrix() { return system_matrix; }
  LA::MPI::Vector & get_system_rhs() { return system_rhs; }
  LA::MPI::Vector & get_system_solution() { return system_solution; }

  const std::map<types::global_dof_index, double> & get_reflective_dJ() const
  {
    return reflective_dJ;
  }
  std::map<types::global_dof_index, double> & get_reflective_dJ() { return reflective_dJ; }
  const std::map<types::global_dof_index, Hat> & get_reflective_dof_normals() const
  {
    return reflective_dof_normals;
  };
  std::vector<std::map<types::global_dof_index, double>> & get_reflective_incoming_flux()
  {
    return reflective_incoming_flux;
  }

  template <typename T>
  static void saveVector(const std::vector<T> & v, const std::string filename)
  {
    std::ofstream f;
    f.open(filename);
    for (unsigned int i = 0; i < v.size(); ++i)
      f << std::scientific << v[i] << std::endl;
    f.close();
  }

private:
  /// Initial setup for the Problem
  void setup();
  /// Primary solver for the Problem
  void solve();

  /// Compute the L2 norm of (scalar_flux - scalar_flux_old) for checking convergence
  double scalar_flux_L2();
  /// Compute the L2 norm of reflective_dJ for checking convergence
  double reflective_dJ_L2();
  /// Build and save .vtu output
  void output_vtu();

  /// MPI communicator
  MPI_Comm comm;
  /// Parallel cout
  ConditionalOStream pcout;
  /// Timer
  TimerOutput timer;

  /// Problem description that holds material properties, boundary conditions, etc
  Description<dim> description;
  /// Problem discretization that holds the dof_handler and triangulation
  Discretization<dim> discretization;
  /// Access to the angular quadrature
  const AngularQuadrature<dim> & aq;
  /// The SNProblem, which computes the SN quantities
  SNProblem<dim> sn;
  /// The DSAProblem, which accelerates the source iteration
  DSAProblem<dim> dsa;

  /// Access to the dof_handler in the Discretization
  const DoFHandler<dim> & dof_handler;

  /// Finite element representation of the scalar flux at the current iteration
  LA::MPI::Vector scalar_flux;
  /// Finite element representation of the scalar flux at the previous iteration
  LA::MPI::Vector scalar_flux_old;
  /// Finite element representation of the angular flux (not stored by default)
  std::vector<LA::MPI::Vector> angular_flux;

  /// Map from a degree of freedom on a reflective boundary to its normal Hat
  std::map<types::global_dof_index, Hat> reflective_dof_normals;
  /// Vector (entry for each direction) of maps from a degree of freedom on a
  /// reflective boundary to its corresponding incoming reflective angular flux
  std::vector<std::map<types::global_dof_index, double>> reflective_incoming_flux;
  /// Map from a degree of freedom on a reflective boundary to its net current
  std::map<types::global_dof_index, double> reflective_dJ;

  /// System storage
  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector system_rhs;
  LA::MPI::Vector system_solution;

  /// Source iteration residuals
  std::vector<double> residuals;

  /// Vtu output filename
  std::string vtu_filename = "output";
  /// Residual output filename
  std::string residual_filename = "";
  /// Bool to save the angular flux
  bool save_angular_flux = false;

  /// Maximum source iterations
  unsigned int max_source_its = 1000;
  /// Source iteration tolerance
  double source_iteration_tol = 1.0e-12;

  /// Maximum reflective boundary iterations
  unsigned int max_ref_its = 1;
  /// Reflective boundary iteration tolerance
  double reflective_iteration_tol = 1.0e-16;

  /// Enable DSA
  bool enable_dsa = true;
};
} // namespace RadProblem

#endif // PROBLEM_H
