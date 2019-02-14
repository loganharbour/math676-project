#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <math.h>

class AngularQuadrature
{
public:
  AngularQuadrature(unsigned int n) : dir(2 * n), w(2 * n, dealii::numbers::PI / n)
  {
    for (unsigned int i = 0; i < 2 * n; ++i)
    {
      double omega = (dealii::numbers::PI_2 + i) / n;
      dir[i][0] = std::cos(omega);
      dir[i][1] = std::sin(omega);
    }
  }

private:
  // Directions
  std::vector<dealii::Point<2>> dir;

  // Weights
  std::vector<double> w;
};

#endif /* ANGULARQUADRATURE_H */
