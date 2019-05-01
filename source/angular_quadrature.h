#ifndef ANGULARQUADRATURE_H
#define ANGULARQUADRATURE_H

#include <boost/algorithm/string.hpp>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <math.h>

namespace RadProblem
{
using namespace dealii;

// Enum for the hat directions, i.e., (1, 0, 0) = XHat, (0, 1, 0) = YHat, ...
enum Hat
{
  InvalidHat = 0,
  XHat = 1,
  NegXHat = 2,
  YHat = 3,
  NegYHat = 4,
  ZHat = 5,
  NegZHat = 6
};

// The angular quadrature types
enum AQType
{
  InvalidAQ = 0,
  ProductAQ = 1,
  TriangularAQ = 2
};

// String to AQType
static inline AQType
string_to_AQType(const std::string & string)
{
  std::string lower_string = string;
  boost::algorithm::to_lower(lower_string);
  if (lower_string == "product" || lower_string == "productaq")
    return ProductAQ;
  if (lower_string == "triangular" || lower_string == "triangularaq")
    return TriangularAQ;
  return InvalidAQ;
}

// Get the Hat from a Tensor<1, dim>
template <int dim>
static Hat
get_hat_direction(const Tensor<1, dim> & v, const double eps = 1e-12)
{
  const Tensor<1, dim> unit_v = v / v.norm();
  if (-std::abs(unit_v[0]) + 1 < eps && std::abs(unit_v[1]) < eps &&
      !(dim == 3 && std::abs(unit_v[2]) < eps))
    return unit_v[0] > 0 ? XHat : NegXHat;
  else if (-std::abs(unit_v[1]) + 1 < eps && std::abs(unit_v[0]) < eps &&
           !(dim == 3 && std::abs(unit_v[2]) < eps))
    return unit_v[1] > 0 ? YHat : NegYHat;
  else if (dim == 3 && -std::abs(unit_v[2]) + 1 < eps && std::abs(unit_v[0]) < eps &&
           std::abs(unit_v[1]) < eps)
    return unit_v[2] > 0 ? ZHat : NegZHat;
  else
    throw ExcMessage("Couldn't find hat direction");
}

// Get the Tensor<1, dim> from a given Hat
template <int dim>
static Tensor<1, dim>
get_hat_direction(const Hat hat)
{
  Tensor<1, dim> v;
  switch (hat)
  {
    case XHat:
      v[0] = 1;
      break;
    case NegXHat:
      v[0] = -1;
      break;
    case YHat:
      v[1] = 1;
      break;
    case NegYHat:
      v[1] = -1;
      break;
    case ZHat:
      v[2] = 1;
      break;
    case NegZHat:
      v[2] = -1;
      break;
    default:
      throw ExcMessage("Unknown Hat in get_hat_direction");
  }

  return v;
}

template <int dim>
class AngularQuadrature
{
public:
  AngularQuadrature() {}

  void init(const unsigned int order, const std::string aq_type_string)
  {
    const AQType aq_type = string_to_AQType(aq_type_string);

    if (initialized)
      throw ExcMessage("The AngularQuadrature object has already been initialized");

    // Product Gauss-Legendre-Chebyshev quadrature for 2D and 3D
    // Nomenclature for SN: N = number of polar levels (positive and negative)
    // Azimuthally, N/2 directions in [0,pi/2]
    // Hence, per octant: N/2 (polar level) times N/2 in azimuth = N^2/4
    if (aq_type == ProductAQ)
      n_directions = (dim == 2 ? 1 : 2) * order * order;
    else if (aq_type == TriangularAQ)
      n_directions = (dim == 2 ? 1 : 2) * order * (order + 2) / 2;
    else
      throw ExcMessage("Unknown angular quadrature type " + aq_type_string);

    const QGauss<1> gq(order);
    const auto & gq_JxW = gq.get_weights();
    const auto & gq_points = gq.get_points();

    // Sum of polar weights = 1 for the entire range in 3D, but 0.5 in 2D
    const double polar_norm = (dim == 2 ? 0.5 : 1);

    // Loop over polar levels (mu = cos(theta))
    for (unsigned int ig = 0; ig < order; ++ig)
    {
      // In 2D, do not use the negative polar levels (which means mu > 0.5
      // when the quadrature is defined over [0,1])
      if (dim == 2 && gq_points[ig](0) <= 0.5)
        continue;

      double n_azi = 2 * order;
      if (aq_type == TriangularAQ)
      {
        // tri shift
        const double tri_shift = -((double)order - 1.) / 2;
        n_azi -= 4 * std::floor(std::abs(tri_shift + ig));
      }
      // Azimuthal weights (Gauss-Chebyshev, thus uniform)
      // Product: wa = 2pi/2N
      // Triangular: wa = 2pi/(4*(N/2-i))
      // Normalize them right away
      double azi_weight = 1. / n_azi;

      const double cos_theta = gq_points[ig](0) * 2.0 - 1.0;
      const double sin_theta = std::sqrt(1 - std::pow(cos_theta, 2));
      const double weight = gq_JxW[ig] * azi_weight / polar_norm;

      // Loop over azimuthal angles (phi in (0,2pi)):
      // Product: 2N azimuthal angles
      //          phi = 2pi/2N*(i+1/2), i = 0, ..., 2N-1
      // Triangular: 2N-4i azimuthal angles
      //             phi = 2pi/(2N-4i)*(i+1/2), i = 0, ..., 2N-4i-1
      for (unsigned int ic = 0; ic < n_azi; ++ic)
      {
        const double phi = ((double)ic + 0.5) * 2 * M_PI / n_azi;

        Tensor<1, dim> direction;
        direction[0] = std::cos(phi) * sin_theta;
        direction[1] = std::sin(phi) * sin_theta;
        if (dim == 3)
          direction[2] = cos_theta;

        // Insert into the quadrature set
        directions.emplace_back(direction);
        weights.emplace_back(weight);
      }
    }

    // Initialize reflected directions (dim * 2 possible reflective normals)
    reflect_to_vector.resize(dim * 2);

    // Loop through all possible hat directions
    // (0 for Hat enum represents invalid, 1 to dim * 2 are valid)
    for (unsigned int hat_i = 1; hat_i <= dim * 2; ++hat_i)
    {
      // This normal direction
      const Hat hat = (Hat)hat_i;
      const Tensor<1, dim> normal = get_hat_direction<dim>((Hat)hat_i);

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
  unsigned int reflect_to(const Hat hat, const unsigned int d_from) const
  {
    return reflect_to_vector[hat].at(d_from);
  }

  // Get the direction that is closest to direction
  unsigned int closest(const Tensor<1, dim> & direction) const
  {
    // Normalize
    const Tensor<1, dim> unit_direction = direction / direction.norm();

    // Start with the maximum value
    double norm_min = std::numeric_limits<double>::max();
    unsigned int d_closest;

    // Check every direction
    for (unsigned int d = 0; d < n_directions; ++d)
    {
      // Check the norm of the difference between this quadrature direction and
      // the direction that is being checked
      const double norm = (directions[d] - unit_direction).norm();
      // Seek the smallest norm of the difference
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
