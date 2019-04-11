#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <math.h>
#include <algorithm>

namespace RadProblem
{
using namespace dealii;

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

      QGauss<1> gq(order);
      const auto & gq_JxW = gq.get_weights();
      const auto & gq_points = gq.get_points();

      for (unsigned int ig = 0; ig < order; ++ig)
      {
        const double cos_phi = gq_points[ig](0) * 2.0 - 1.0;
        const double sin_phi = std::sin(std::acos(cos_phi));

        for (unsigned int ic = 0; ic < order * 2; ++ic)
        {
          const double w = (2 * (double)(ic + 1) - 1) * M_PI / n_per_quad;

          Tensor<1, dim> direction;
          direction[0] = std::cos(w) * sin_phi;
          direction[1] = std::sin(w) * sin_phi;
          direction[2] = cos_phi;
          directions.emplace_back(direction);

          weights.emplace_back(gq_JxW[ig] / n_per_quad);
        }
      }
    }

    init_reflected_directions();
  }

  void init_reflected_directions()
  {
    reflected_directions.resize(dim, std::vector<unsigned int>(n_directions));

    // 0 is ehat_x, 1 is ehat_y, 2 is ehat_z
    std::vector<dealii::Tensor<1, dim>> normals(dim);
    normals[0][0] = 1;
    normals[1][0] = 0;
    normals[0][1] = 0;
    normals[1][1] = 1;
    if (dim == 3)
    {
      normals[2][0] = 0;
      normals[2][1] = 0;
      normals[0][2] = 0;
      normals[1][2] = 0;
      normals[2][2] = 1;
    }

    // Fill for each ehat direction
    for (unsigned int i = 0; i < dim; ++i)
    {
      // Find the incoming direction for each outgoing direction
      for (unsigned int d = 0; d < n_directions; ++d)
      {
        // The reflected (incoming) direction for this outgoing direction
        const Tensor<1, dim> dir_ref =
            directions[d] - 2 * (directions[d] * normals[i]) * normals[i];

        // Try each incoming direction. Pick the one with the smallest L2
        // difference of the reflected direction and quadrature direction
        unsigned int d_min;
        double norm_min = std::numeric_limits<double>::max();
        for (unsigned int dp = 0; dp < n_directions; ++dp)
        {
          double norm = (directions[dp] - dir_ref).norm();
          if (norm < norm_min)
          {
            d_min = dp;
            norm_min = norm;
          }
        }
        reflected_directions[i][d] = d_min;
      }
    }
  }

  unsigned int reflected_d(const unsigned int d, const Tensor<1, dim> & normal) const
  {
    double eps = 1e-12;
    if (std::abs(normal[0]) - 1 < eps)
      return reflected_directions[0][d];
    else if (std::abs(normal[1]) - 1 < eps)
      return reflected_directions[1][d];
    else if (dim == 3 && std::abs(normal[2]) - 1 < eps)
      return reflected_directions[2][d];
    else
      throw ExcMessage("Couldn't find reflected directon");
  }

  unsigned int n_dir() const { return n_directions; }
  Tensor<1, dim> dir(const unsigned int d) const { return directions[d]; }
  double w(const unsigned int d) const { return weights[d]; }

private:
  /// Number of directions
  unsigned int n_directions;

  /// Angular quadrature directions
  std::vector<Tensor<1, dim>> directions;

  /// Reflected directions
  std::vector<std::vector<unsigned int>> reflected_directions;

  /// Angular quadrature weights
  std::vector<double> weights;
};
} // namespace RadProblem

#endif /* ANGULARQUADRATURE_H */
