#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>

#include <math.h>

namespace SNProblem
{
class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(unsigned int n)
  {
    directions.resize(2 * n);
    weights.resize(2 * n, 1.0 / (2 * n));
    for (unsigned int i = 0; i < 2 * n; ++i)
    {
      double omega = dealii::numbers::PI * (i + 0.5) / n;
      directions[i][0] = std::cos(omega);
      directions[i][1] = std::sin(omega);
    }
  }

  unsigned int n_dir() const { return directions.size(); }
  dealii::Tensor<1, 2> dir(unsigned int i) const { return directions[i]; }
  double w(unsigned int i) const { return weights[i]; }

private:
  /// Quadrature directions
  std::vector<dealii::Tensor<1, 2>> directions;

  /// Quadrature weights
  std::vector<double> weights;
};
} // namespace SNProblem

#endif /* ANGULARQUADRATURE_H */
