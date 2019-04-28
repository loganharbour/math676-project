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

    // Product Gauss-Legendre-Chebyshev quadrature for 2D and 3D
    // Nomenclature: SN: N=number of polar levels (positive and negative)
    // Azimuthally, N/2 direction in [0,pi/2]
    // Hence, per octant: N/2 (polar level) times N/2 in azimuth = N^2/4
    if (dim == 2)
      n_directions = order * order;
    else if (dim==3)
      n_directions = 2 * order * order;
    std::cout << "nd dir = " << n_directions << std::endl;

    directions.resize(n_directions);
    weights.resize(n_directions);

    QGauss<1> gq(order);
    const auto & gq_JxW = gq.get_weights();
    const auto & gq_points = gq.get_points();

    double sum_weights = 0.;
    double sum_polar_weights = 0.;
    for (unsigned int ig = 0; ig < order; ++ig)
       sum_polar_weights += gq_JxW[ig];
    std::cout << "sum polar weights " << sum_polar_weights << std::endl;
    /* this is not working. why ??
    std::cout << "sum polar weights = "
              << std::accumulate(gq_JxW.begin(), gq_JxW.end(), 0) << std::endl;
    */

    // azimuthal weights (Gauss-Chebyshev, thus uniform)
    // wa = 2pi/2N = pi/N. We normlaize them right away:
    const double azi_weight = 1. / ( 2. * order );
    // furthermore, sum of polar weights = 1 for the entire range
    // however, if in 2D, their sum is only equal to 0.5
    double polar_norm = 1.;
    if (dim==2)
      polar_norm = 0.5;

    unsigned int d=0;
    // loop over polar levels (mu=cos(theta))
    for (unsigned int ig = 0; ig < order; ++ig)
    {
      // in 2D, we do not use the negative polar levels (which means mu>0.5
      // when the quadrature is defined over [0,1])
      if ( (dim==3) || ((dim==2) && (gq_points[ig](0)>0.5)) )
      {
        const double cos_theta = gq_points[ig](0) * 2.0 - 1.0;
        const double sin_theta = std::sqrt(1-std::pow(cos_theta,2));

        // loop over azimuthal angles (phi in (0,2pi). 2N azimuthal angles
        // phi = 2pi/2N*(i+1/2), i=0,...,2N-1
        for (unsigned int ic = 0; ic < 2 * order; ++ic)
        {
          const double phi = ( (double)ic + 0.5 ) * M_PI / order;

          Tensor<1, dim> direction;
          direction[0] = std::cos(phi) * sin_theta;
          direction[1] = std::sin(phi) * sin_theta;
          if (dim==3)
            direction[2] = cos_theta;
          directions.emplace_back(direction);

          weights.emplace_back(gq_JxW[ig] * azi_weight);
          // printout
          sum_weights += gq_JxW[ig] / polar_norm * azi_weight;
          std::cout << "d= " << d << " ig=" << ig << " ic=" << ic << " , "
                    << cos_theta << " , " << sin_theta << " , " << azi_weight
                    << " , " << direction[0] << " , " << direction[1] << std::endl;
        }
      }
      ++d;
    }
    // printout
    for (unsigned int d =0; d < n_directions; ++d)
    {
      std::cout << directions[d][0] << " , " << directions[d][1];
      if (dim==3)
        std::cout << " , " << directions[d][2];
      std::cout << " , " << weights[d] << std::endl;
    }
    std::cout << "sum weights = "
              << std::accumulate(weights.begin(), weights.end(), 0)
              << " other sum = " << sum_weights << std::endl;

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
