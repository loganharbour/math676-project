#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <math.h>
#include <algorithm>

namespace RadProblem
{

template <int dim>
class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(const unsigned int order)
  {
    if (dim == 2)
    {
      n_directions = order * 2;

      directions.resize(n_directions);
      weights.resize(n_directions, 1.0 / (double)n_directions);

      for (unsigned int id = 0; id < n_directions; ++id)
      {
        const double w = (2 * (double)(id + 1) - 1) * M_PI / (double)n_directions;
        directions[id][0] = std::cos(w);
        directions[id][1] = std::sin(w);
      }
    }
    else if (dim == 3)
    {
      n_directions = order * 8;
      const double n_per_quad = (double)order * 2;

      dealii::QGauss<1> gq(order);
      const auto & gq_JxW = gq.get_weights();
      const auto & gq_points = gq.get_points();

      for (unsigned int ig = 0; ig < order; ++ig)
      {
        const double cos_phi = gq_points[ig](0) * 2.0 - 1.0;
        const double sin_phi = std::sin(std::acos(cos_phi));

        for (unsigned int ic = 0; ic < order * 2; ++ic)
        {
          const double w = (2 * (double)(ic + 1) - 1) * M_PI / n_per_quad;

          dealii::Tensor<1, dim> direction;
          direction[0] = std::cos(w) * sin_phi;
          direction[1] = std::sin(w) * sin_phi;
          direction[2] = cos_phi;
          directions.emplace_back(direction);

          weights.emplace_back(gq_JxW[ig] / n_per_quad);
        }
      }
    }
  }

  unsigned int n_dir() const { return n_directions; }
  dealii::Tensor<1, dim> dir(const unsigned int d) const { return directions[d]; }
  double w(const unsigned int d) const { return weights[d]; }

private:
  /// Number of directions
  unsigned int n_directions;

  /// Angular quadrature directions
  std::vector<dealii::Tensor<1, dim>> directions;

  /// Angular quadrature weights
  std::vector<double> weights;
};
} // namespace RadProblem

#endif /* ANGULARQUADRATURE_H */
