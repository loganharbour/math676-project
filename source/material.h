#ifndef MATERIAL_H
#define MATERIAL_H

namespace RadProblem
{
class Material
{
public:
  Material(const double sigma_t, const double sigma_s, const double src)
    : D(1.0 / (3 * sigma_t)),
      sigma_t(sigma_t),
      sigma_s(sigma_s),
      sigma_a(sigma_t - sigma_s),
      src(src)
  {
  }
  Material(const Material & material)
    : D(material.D),
      sigma_t(material.sigma_t),
      sigma_s(material.sigma_s),
      sigma_a(material.sigma_a),
      src(material.src)
  {
  }

  // Diffusion coefficient [cm]
  const double D;
  // Macroscopic total cross section [1/cm]
  const double sigma_t;
  // Macroscopic scattering cross section [1/cm]
  const double sigma_s;
  // Macroscopic absorption cross section [1/cm]
  const double sigma_a;
  // Volumetric source term [p/cm^3]
  const double src;
};
} // namespace RadProblem

#endif /* MATERIAL_H */
