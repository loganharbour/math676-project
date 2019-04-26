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

enum HatDirection
{
  X = 0,
  NEG_X = 1,
  Y = 2,
  NEG_Y = 3,
  Z = 4,
  NEG_Z = 5
};

template <int dim>
static HatDirection
get_hat_direction(const Tensor<1, dim> & v, const double eps = 1e-12)
{
  if (-std::abs(v[0]) + 1 < eps)
    if (v[0] > 0)
      return HatDirection::X;
    else
      return HatDirection::NEG_X;
  else if (-std::abs(v[1]) + 1 < eps)
    if (v[1] > 0)
      return HatDirection::Y;
    else
      return HatDirection::NEG_Y;
  else if (dim == 3 && -std::abs(v[2]) + 1 < eps)
    if (v[2] > 0)
      return HatDirection::Z;
    else
      return HatDirection::NEG_Z;
  else
    throw ExcMessage("Couldn't find hat direction");
}

template <int dim>
static dealii::Tensor<1, dim>
get_hat_direction(const HatDirection hat_direction)
{
  Tensor<1, dim> v;
  if (dim == 2)
  {
    v[0] = 0;
    v[1] = 0;
  }
  if (dim == 3)
    v[2] = 0;

  switch (hat_direction)
  {
    case X:
      v[0] = 1;
      break;
    case NEG_X:
      v[0] = -1;
      break;
    case Y:
      v[1] = 1;
      break;
    case NEG_Y:
      v[1] = -1;
      break;
    case Z:
      v[2] = 1;
      break;
    case NEG_Z:
      v[2] = -1;
      break;
  }

  return v;
}

template <int dim>
class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(const unsigned int order)
  {
    if (initialized)
      throw ExcMessage("The AngularQuadrature object has already been initialized");

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

    // Initialize reflected directions (dim * 2 possible reflective normals)
    reflect_to_vector.resize(dim * 2);
    for (unsigned int hat_i = 0; hat_i < dim * 2; ++hat_i)
    {
      // This normal direction
      const HatDirection hat = (HatDirection)hat_i;
      const Tensor<1, dim> normal = get_hat_direction<dim>((HatDirection)hat_i);

      // Check every outgoing direction to store what direction it goes into
      for (unsigned int d_from = 0; d_from < n_directions; ++d_from)
      {
        // Direction is not outgoing
        if (directions[d_from] * normal < 0)
          continue;

        // The direction that this direction reflects into
        const Tensor<1, dim> dir_ref = directions[d_from] - 2 * (directions[d_from] * normal) * normal;

        // Find the direction in the quadrature set most similar to dir_ref and store
        const unsigned int d_to = closest(dir_ref);
        reflect_to_vector[hat].emplace(d_from, d_to);
      }
    }
  }

  // Get the direction that direction d_from reflects into on the surface defined by normal
  unsigned int reflect_to(const HatDirection hat, const unsigned int d_from) const
  {
    return reflect_to_vector[hat].at(d_from);
  }

  // Get the direction that is closest to direction
  unsigned int closest(const Tensor<1, dim> & direction) const
  {
    unsigned int d_closest;
    double norm_min = std::numeric_limits<double>::max();
    for (unsigned int d = 0; d < n_directions; ++d)
    {
      double norm = (directions[d] - direction).norm();
      if (norm < norm_min)
      {
        d_closest = d;
        norm_min = norm;
      }
    }

    return d_closest;
  }

  unsigned int n_dir() const { return n_directions; }
  Tensor<1, dim> dir(const unsigned int d) const { return directions[d]; }
  double w(const unsigned int d) const { return weights[d]; }

private:
  /// Whether or not the aq object is initialized
  bool initialized = false;

  /// Number of directions
  unsigned int n_directions;

  /// Angular quadrature directions
  std::vector<Tensor<1, dim>> directions;

  /// Reflected directions, sorted by the normal directions
  std::vector<std::map<unsigned int, unsigned int>> reflect_to_vector;

  /// Angular quadrature weights
  std::vector<double> weights;
};
} // namespace RadProblem

#endif /* ANGULARQUADRATURE_H */
