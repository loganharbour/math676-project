#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

DeclException1(MaterialExists, int, "Material with id " << arg1 << " already exists.");

class Material
{
public:
  Material(const double sigma_t, const double sigma_s, const double src)
    : sigma_s(sigma_s), sigma_t(sigma_t), src(src)
  {
  }
  Material(const Material & material)
    : sigma_s(material.sigma_s), sigma_t(material.sigma_t), src(material.src)
  {
  }

  // Macroscopic scattering cross section [1/cm]
  const double sigma_s;
  // Macroscopic total cross section [1/cm]
  const double sigma_t;
  // Volumetric source term [p/cm^3]
  const double src;
};

#endif /* MATERIAL_H */
