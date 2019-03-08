#ifndef DESCRIPTION_H
#define DESCRIPTION_H

#include <deal.II/base/parameter_acceptor.h>
#include <map>

namespace SNProblem
{
// Forward declaration
class Material;

class Description : public dealii::ParameterAcceptor
{
public:
  Description();

  void setup();

  const std::map<const unsigned int, const Material> & get_materials() const { return materials; }

private:
  void setup_materials();
  void setup_boundary_conditions();

  std::map<const unsigned int, const Material> materials;

  // Boundary conditions
  std::map<const unsigned int, const double> perpendicular_bcs;
  std::map<const unsigned int, const double> isotropic_bcs;

  std::vector<unsigned int> material_ids;
  std::vector<double> material_sigma_t;
  std::vector<double> material_sigma_s;
  std::vector<double> material_src;

  // Input perpendicular incident flux boundary conditions
  std::vector<unsigned int> perpendicular_boundary_ids;
  std::vector<double> perpendicular_boundary_fluxes;
  // Input isotropic incident flux boundary conditions
  std::vector<unsigned int> isotropic_boundary_ids;
  std::vector<double> isotropic_boundary_fluxes;
};
} // namespace SNProblem

#endif // DESCRIPTION_H
