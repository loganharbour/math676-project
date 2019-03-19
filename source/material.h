#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

namespace RadProblem
{
class Material
{
public:
  Material(const double sigma_t, const double sigma_s, const double src)
    : sigma_t(sigma_t), sigma_s(sigma_s), src(src)
  {
  }
  Material(const Material & material)
    : sigma_t(material.sigma_t), sigma_s(material.sigma_s), src(material.src)
  {
  }

  // Macroscopic total cross section [1/cm]
  const double sigma_t;
  // Macroscopic scattering cross section [1/cm]
  const double sigma_s;
  // Volumetric source term [p/cm^3]
  const double src;
};
} // namespace RadProblem

#endif /* MATERIAL_H */
