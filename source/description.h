#ifndef DESCRIPTION_H
#define DESCRIPTION_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <map>

namespace RadProblem
{

template <int dim>
class AngularQuadrature;
template <int dim>
class Discretization;

// The possible boundary condition types
enum BCTypes
{
  Isotropic,
  Incident,
  Reflective,
  Vacuum
};

// Struct for boundary condition storage
struct BC
{
  BC(const BCTypes type, const double value = 0, const unsigned int d = 0)
    : type(type), value(value), d(d)
  {
  }
  // The boundary condition type
  const BCTypes type;
  // The boundary condition value (if applicable)
  const double value;
  // The boundary condition direction (if applicable)
  const unsigned int d;
};

// Struct for material storage
struct Material
{
  Material(const double sigma_t, const double sigma_s, const double src)
    : D(1.0 / (3 * std::max(sigma_t, 1E-5))),
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

template <int dim>
class Description : public dealii::ParameterAcceptor
{
public:
  Description();

  // Setup to be called by the Problem
  void setup(const Discretization<dim> & discretization);

  // Get the bc for boundary_id
  const BC & get_bc(const unsigned int boundary_id) const;
  // Whether or not the problem has incident bcs
  bool has_incident_bcs() const { return incident_bcs; }
  // Whether or not the problem has reflecting bcs
  bool has_reflecting_bcs() const { return reflecting_bcs; }

  // Get the material for material_id
  const Material & get_material(const unsigned int material_id) const;
  // Whether or not the problem has scattering
  bool has_scattering() const { return scattering; }

private:
  void fill_bcs(const BCTypes type,
                const std::vector<unsigned int> & ids,
                const std::vector<double> * values = NULL,
                const std::vector<double> * directions = NULL,
                const AngularQuadrature<dim> * aq = NULL);

  void setup_bcs(const std::set<unsigned int> & mesh_boundary_ids,
                 const AngularQuadrature<dim> & aq);
  void setup_materials(const std::set<unsigned int> & mesh_material_ids);

  // Material storage
  std::map<const unsigned int, const Material> materials;
  // Whether or not the problem has scattering
  bool scattering = false;

  // Boundary condition storage
  std::map<const unsigned int, const BC> bcs;
  // Whether or not we have incident boundary conditions
  bool incident_bcs = false;
  // Whether or not we have reflecting boundary conditions
  bool reflecting_bcs = false;

  // Input materials
  std::vector<unsigned int> material_ids = {};
  std::vector<double> material_sigma_t = {};
  std::vector<double> material_sigma_s = {};
  std::vector<double> material_src = {};

  // Input vacuum boundary ids
  std::vector<unsigned int> vacuum_boundary_ids = {};
  // Input isotropic incident flux boundary conditions
  std::vector<unsigned int> isotropic_boundary_ids = {};
  std::vector<double> isotropic_boundary_fluxes = {};
  // Input incident flux boundary conditions
  std::vector<unsigned int> incident_boundary_ids = {};
  std::vector<double> incident_boundary_fluxes = {};
  std::vector<double> incident_boundary_directions = {};
  // Input reflective boundary conditions
  std::vector<unsigned int> reflective_boundary_ids = {};
};
} // namespace RadProblem

#endif // DESCRIPTION_H
