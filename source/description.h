#ifndef DESCRIPTION_H
#define DESCRIPTION_H

#include <deal.II/base/parameter_acceptor.h>

#include <map>

namespace RadProblem
{
// Forward declaration
class Material;

class Description : public dealii::ParameterAcceptor
{
public:
  Description();

  void setup(const std::set<unsigned int> & mesh_material_ids);

  const std::map<const unsigned int, const Material> & get_materials() const { return materials; }
  const std::map<const unsigned int, const double> & get_perpendicular_bcs() const
  {
    return perpendicular_bcs;
  }
  const std::map<const unsigned int, const double> & get_isotropic_bcs() const
  {
    return isotropic_bcs;
  }
  bool has_incident_bcs() const { return incident_bcs; }
  bool has_scattering() const { return scattering; }

private:
  void setup_materials(const std::set<unsigned int> & mesh_material_ids);
  void setup_boundary_conditions();

  std::map<const unsigned int, const Material> materials;

  // Boundary conditions
  std::map<const unsigned int, const double> perpendicular_bcs;
  std::map<const unsigned int, const double> isotropic_bcs;
  bool incident_bcs = false;

  std::vector<unsigned int> material_ids = {};
  std::vector<double> material_sigma_t = {};
  std::vector<double> material_sigma_s = {};
  std::vector<double> material_src = {};
  bool scattering = false;

  // Input perpendicular incident flux boundary conditions
  std::vector<unsigned int> perpendicular_boundary_ids = {};
  std::vector<double> perpendicular_boundary_fluxes = {};
  // Input isotropic incident flux boundary conditions
  std::vector<unsigned int> isotropic_boundary_ids = {};
  std::vector<double> isotropic_boundary_fluxes = {};
};
} // namespace RadProblem

#endif // DESCRIPTION_H
