#ifndef DESCRIPTION_H
#define DESCRIPTION_H

#include <deal.II/base/parameter_acceptor.h>

#include <map>

namespace RadProblem
{
class Description : public dealii::ParameterAcceptor
{
public:
  Description();

  // The possible boundary condition types
  enum BCTypes
  {
    Isotropic,
    Perpendicular,
    Reflective,
    Vacuum
  };

  // Struct for boundary condition storage
  struct BC
  {
    BC(const BCTypes type, const double value = 0) : type(type), value(value) {}
    // The boundary condition type
    const BCTypes type;
    // The boundary condition value (if applicable)
    const double value;
  };

  // Struct for material storage
  struct Material
  {
    Material(const double sigma_t, const double sigma_s, const double src)
      : D(1.0 / (3 * sigma_t)),
        sigma_t(sigma_t),
        sigma_s(sigma_s),
        sigma_a(sigma_t - sigma_s),
        src(src)
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

  // Setup to be called by the Problem
  void setup(const std::set<unsigned int> & mesh_boundary_ids,
             const std::set<unsigned int> & mesh_material_ids);

  // Get the bc for boundary_id
  const BC & get_bc(const unsigned int boundary_id) const;
  // Whether or not the problem has incident bcs
  bool has_incident_bcs() const { return incident_bcs; }

  // Get the material for material_id
  const Material & get_material(const unsigned int material_id) const;
  // Whether or not the problem has scattering
  bool has_scattering() const { return scattering; }

private:
  void fill_bcs(const BCTypes type,
                const std::vector<unsigned int> & ids,
                const std::vector<double> * values = NULL);

  void setup_bcs(const std::set<unsigned int> & mesh_boundary_ids);
  void setup_materials(const std::set<unsigned int> & mesh_material_ids);

  // Material storage
  std::map<const unsigned int, const Material> materials;
  // Whether or not the problem has scattering
  bool scattering = false;

  // Boundary condition storage
  std::map<const unsigned int, const BC> bcs;
  // The unique boundarty ids input by the user
  std::set<unsigned int> input_bc_ids;
  // Whether or not we have incident boundary conditions
  bool incident_bcs = false;

  // Input materials
  std::vector<unsigned int> material_ids = {};
  std::vector<double> material_sigma_t = {};
  std::vector<double> material_sigma_s = {};
  std::vector<double> material_src = {};

  // Input vacuum boundary ids
  std::vector<unsigned int> vacuum_boundary_ids = {0};
  // Input perpendicular incident flux boundary conditions
  std::vector<unsigned int> perpendicular_boundary_ids = {};
  std::vector<double> perpendicular_boundary_fluxes = {};
  // Input isotropic incident flux boundary conditions
  std::vector<unsigned int> isotropic_boundary_ids = {};
  std::vector<double> isotropic_boundary_fluxes = {};
  // Input reflective boundary conditions
  std::vector<unsigned int> reflective_boundary_ids = {};
};
} // namespace RadProblem

#endif // DESCRIPTION_H
