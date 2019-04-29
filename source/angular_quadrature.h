#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <math.h>

namespace RadProblem
{
using namespace dealii;

// Enum for the hat directions, i.e., (1, 0, 0) = X, (0, 1, 0) = Y, ...
enum HatDirection
{
  X = 0,
  NEG_X = 1,
  Y = 2,
  NEG_Y = 3,
  Z = 4,
  NEG_Z = 5
};

// Get the HatDirection from a Tensor<1, dim>
template <int dim>
static HatDirection
get_hat_direction(const Tensor<1, dim> & v, const double eps = 1e-12)
{
  const Tensor<1, dim> unit_v = v / v.norm();
  // Doesn't currently check if the Tensor has zeros in the other dimension
  if (-std::abs(unit_v[0]) + 1 < eps)
    return unit_v[0] > 0 ? HatDirection::X : HatDirection::NEG_X;
  else if (-std::abs(unit_v[1]) + 1 < eps)
    return unit_v[1] > 0 ? HatDirection::Y : HatDirection::NEG_Y;
  else if (dim == 3 && -std::abs(unit_v[2]) + 1 < eps)
    return unit_v[2] > 0 ? HatDirection::Z : HatDirection::NEG_Z;
  else
    throw ExcMessage("Couldn't find hat direction");
}

// Get the Tensor<1, dim> from a given HatDirection
template <int dim>
static Tensor<1, dim>
get_hat_direction(const HatDirection hat_direction)
{
  Tensor<1, dim> v;
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

    // Product Gauss-Legendre-Chebyshev quadrature for 2D and 3D
    // Nomenclature for SN: N = number of polar levels (positive and negative)
    // Azimuthally, N/2 directions in [0,pi/2]
    // Hence, per octant: N/2 (polar level) times N/2 in azimuth = N^2/4
    n_directions = (dim == 2 ? 1 : 2) * order * order;

    QGauss<1> gq(order);
    const auto & gq_JxW = gq.get_weights();
    const auto & gq_points = gq.get_points();

    // Azimuthal weights (Gauss-Chebyshev, thus uniform)
    // wa = 2pi/2N = pi/N; normalize them right away
    const double azi_weight = 1. / (2. * order);
    // Sum of polar weights = 1 for the entire range in 3D, but 0.5 in 2D
    const double polar_norm = (dim == 2 ? 0.5 : 1);

    // Loop over polar levels (mu = cos(theta))
    for (unsigned int ig = 0; ig < order; ++ig)
    {
      // In 2D, do not use the negative polar levels (which means mu > 0.5
      // when the quadrature is defined over [0,1])
      if (dim == 2 && gq_points[ig](0) <= 0.5)
        continue;

      const double cos_theta = gq_points[ig](0) * 2.0 - 1.0;
      const double sin_theta = std::sqrt(1 - std::pow(cos_theta, 2));

      // Loop over azimuthal angles (phi in (0,2pi)). 2N azimuthal angles
      // phi = 2pi/2N*(i+1/2), i = 0, ..., 2N-1
      for (unsigned int ic = 0; ic < 2 * order; ++ic)
      {
        const double phi = ((double)ic + 0.5) * M_PI / order;

        Tensor<1, dim> direction;
        direction[0] = std::cos(phi) * sin_theta;
        direction[1] = std::sin(phi) * sin_theta;
        if (dim == 3)
          direction[2] = cos_theta;
        directions.emplace_back(direction);

        weights.emplace_back(gq_JxW[ig] * azi_weight / polar_norm);
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
        const Tensor<1, dim> dir_ref =
            directions[d_from] - 2 * (directions[d_from] * normal) * normal;

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
    const Tensor<1, dim> unit_direction = direction / direction.norm();
    unsigned int d_closest;
    double norm_min = std::numeric_limits<double>::max();
    for (unsigned int d = 0; d < n_directions; ++d)
    {
      double norm = (directions[d] - unit_direction).norm();
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
