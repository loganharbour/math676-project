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
    N = 2 * n;
    directions.resize(N);
    weights.resize(N, 1.0 / N);
    quadrants.resize(N);
    for (unsigned int d = 0; d < N; ++d)
    {
      double omega = dealii::numbers::PI * (d + 0.5) / n;
      directions[d][0] = std::cos(omega);
      directions[d][1] = std::sin(omega);
      quadrants[d] = (unsigned int)std::floor((float)d * 4 / N);
    }
  }

  unsigned int n_dir() const { return N; }
  dealii::Tensor<1, 2> dir(const unsigned int d) const { return directions[d]; }
  double w(const unsigned int d) const { return weights[d]; }
  unsigned int quadrant(const unsigned int d) const { return quadrants[d]; }

private:
  // Number of directions
  unsigned int N;

  /// Quadrature directions
  std::vector<dealii::Tensor<1, 2>> directions;

  /// Quadrature weights
  std::vector<double> weights;

  /// Quadrants
  std::vector<unsigned int> quadrants;
};
} // namespace SNProblem

#endif /* ANGULARQUADRATURE_H */
