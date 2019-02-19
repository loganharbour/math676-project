#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <math.h>

class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(unsigned int n)
  {
    directions.resize(2 * n);
    weights.resize(2 * n, dealii::numbers::PI / n);
    for (unsigned int i = 0; i < 2 * n; ++i)
    {
      double omega = (dealii::numbers::PI_2 + i) / n;
      directions[i][0] = std::cos(omega);
      directions[i][1] = std::sin(omega);
    }
  }

  unsigned int n_dir() { return directions.size(); }
  dealii::Point<2> dir(unsigned int i) { return directions[i]; }
  double w(unsigned int i) { return weights[i]; }

private:
  // Directions
  std::vector<dealii::Point<2>> directions;

  // Weights
  std::vector<double> weights;
};

#endif /* ANGULARQUADRATURE_H */
