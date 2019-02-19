#ifndef MATERIAL_H
#define MATERIAL_H

#include <iostream>

class Material
{
public:
  Material(double sig_t, double sig_s, double q) : sig_t(sig_t), sig_s(sig_s), q(q) {}
  Material(const Material & material) : sig_t(material.sig_t), sig_s(material.sig_s), q(material.q)
  {
  }

  // Macroscopic total cross section [1/cm]
  const double sig_t;
  // Macroscopic scattering cross section [1/cm]
  const double sig_s;
  // Volumetric source term [p/cm^3]
  const double q;
};

#endif /* MATERIAL_H */
