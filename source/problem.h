#ifndef PROBLEM_H
#define PROBLEM_H

#include "description.h"
#include "discretization.h"
#include "dsaproblem.h"
#include "snproblem.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/vector.h>

#include <fstream>

namespace RadProblem
{
using namespace dealii;

// Forward declarations
class AngularQuadrature;
class Material;

class Problem : public ParameterAcceptor
{
public:
  Problem();

  void run();

  const Description & get_description() const { return description; }
  Discretization & get_discretization() { return discretization; }
  const Discretization & get_discretization() const { return discretization; }

  Vector<double> & get_scalar_flux() { return scalar_flux; }
  Vector<double> & get_scalar_flux_old() { return scalar_flux_old; }
  const Vector<double> & get_scalar_flux_old() const { return scalar_flux_old; }

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
  double scalar_flux_L2() const;
  /// Build and save .vtu output
  void output_vtu() const;

  /// Problem description that holds material properties, boundary conditions, etc
  Description description;
  /// Problem discretization that holds the dof_handler and triangulation
  Discretization discretization;
  /// The SNProblem, which computes the SN quantities
  SNProblem sn;
  /// The DSAProblem, which accelerates the source iteration
  DSAProblem dsa;

  /// Access to the dof_handler in the Discretization
  const DoFHandler<2> & dof_handler;

  /// Finite element representation of the scalar flux at the current iteration
  Vector<double> scalar_flux;
  /// Finite element representation of the scalar flux at the previous iteration
  Vector<double> scalar_flux_old;

  /// Source iteration residuals
  std::vector<double> residuals;

  /// Vtu output filename
  std::string vtu_filename = "output";
  /// Residual output filename
  std::string residual_filename = "";
  /// Maximum source iterations
  unsigned int max_its = 1000;
  /// Source iteration tolerance
  double source_iteration_tol = 1.0e-12;
};
} // namespace RadProblem

#endif // PROBLEM_H
