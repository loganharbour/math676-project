#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>

#include <math.h>
#include <algorithm>

namespace RadProblem
{
class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(unsigned int n)
  {
    N = 2 * n;
    directions.resize(N);
    weights.resize(N, 1.0 / N);
    for (unsigned int d = 0; d < N; ++d)
    {
      double omega = dealii::numbers::PI * (d + 0.5) / n;
      directions[d][0] = std::cos(omega);
      directions[d][1] = std::sin(omega);
    }
    for (unsigned int d = 0; d < n / 2; ++d)
    {
      std::swap(directions[d], directions[d + 3 * n / 2]);
      std::swap(weights[d], weights[d + 3 * n / 2]);
    }
  }

  unsigned int n_dir() const { return N; }
  dealii::Tensor<1, 2> dir(const unsigned int d) const { return directions[d]; }
  double w(const unsigned int d) const { return weights[d]; }

private:
  /// Number of directions
  unsigned int N;

  /// Quadrature directions
  std::vector<dealii::Tensor<1, 2>> directions;

  /// Quadrature weights
  std::vector<double> weights;
};
} // namespace RadProblem

#endif /* ANGULARQUADRATURE_H */
