#ifndef PROBLEM_H
#define PROBLEM_H

#include "description.h"
#include "discretization.h"
#include "material.h"
#include "snproblem.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

#include <map>

namespace RadProblem
{
using namespace dealii;

// Forward declarations
class AngularQuadrature;

class Problem : public ParameterAcceptor
{
public:
  Problem();

  void run();

  const Description & get_description() const { return description; }
  Discretization & get_discretization() { return discretization; }
  Vector<double> & get_scalar_flux() { return scalar_flux; }
  Vector<double> & get_scalar_flux_old() { return scalar_flux_old; }

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
  void setup();

  void solve();

  void postprocess() const;
  void output_vtu() const;

  Description description;
  Discretization discretization;

  const DoFHandler<2> & dof_handler;
  const std::map<const unsigned int, const Material> & materials;
  const AngularQuadrature & aq;
  SNProblem sn;

  /// Finite element representation of the scalar flux at the current iteration
  Vector<double> scalar_flux;
  /// Finite element representation of the scalar flux at the previous iteration
  Vector<double> scalar_flux_old;

  /// Vtu output filename
  std::string vtu_filename = "output";
};
} // namespace RadProblem

#endif // PROBLEM_H
