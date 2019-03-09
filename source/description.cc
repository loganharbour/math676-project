#include "description.h"
#include "material.h"

#include <algorithm>
#include <deal.II/base/exceptions.h>

namespace SNProblem
{
using namespace dealii;

Description::Description() : ParameterAcceptor("Description")
{
  // Initialize default material properties
  material_ids = {0};
  material_sigma_t = {100.0};
  material_sigma_s = {0.0};
  material_src = {1.0};
  // Initialize default boundary conditions
  perpendicular_boundary_ids = {};
  perpendicular_boundary_fluxes = {};
  isotropic_boundary_ids = {};
  isotropic_boundary_fluxes = {};

  // Add material parameters
  add_parameter("material_ids", material_ids);
  add_parameter("material_sigma_t", material_sigma_t);
  add_parameter("material_sigma_s", material_sigma_s);
  add_parameter("material_src", material_src);
  // Add boundary parameters
  add_parameter("perpendicular_boundary_ids", perpendicular_boundary_ids);
  add_parameter("perpendicular_boundary_fluxes", perpendicular_boundary_fluxes);
  add_parameter("isotropic_boundary_ids", isotropic_boundary_ids);
  add_parameter("isotropic_boundary_fluxes", isotropic_boundary_fluxes);
}

void
Description::setup()
{
  setup_materials();
  setup_boundary_conditions();
}

void
Description::setup_materials()
{
  // Make sure all material vectors are the same size
  const auto & double_inputs = {material_sigma_t, material_sigma_s, material_src};
  for (const auto & input : double_inputs)
    if (input.size() != material_ids.size())
      throw ExcMessage("Material input size mismatch");

  // Create a Material for each id
  for (unsigned int i = 0; i < material_ids.size(); ++i)
  {
    if (material_sigma_s[i] > 0)
      scattering = true;
    if (materials.find(material_ids[i]) != materials.end())
      throw ExcMessage("Material id already exists in material map");
    materials.emplace(material_ids[i],
                      Material(material_sigma_t[i], material_sigma_s[i], material_src[i]));
  }
}

void
Description::setup_boundary_conditions()
{
  // Fill perpendicular boundary conditions
  if (perpendicular_boundary_ids.size() != perpendicular_boundary_fluxes.size())
    throw ExcMessage("Perpendicular boundary inputs are not of the same size");
  for (unsigned int i = 0; i < perpendicular_boundary_ids.size(); ++i)
  {
    if (perpendicular_bcs.find(perpendicular_boundary_ids[i]) != perpendicular_bcs.end())
      throw ExcMessage("Duplicate perpendicular boundary ids provided");
    perpendicular_bcs.emplace(perpendicular_boundary_ids[i], perpendicular_boundary_fluxes[i]);
  }

  // Fill isotropic boundary conditions
  if (isotropic_boundary_ids.size() != isotropic_boundary_fluxes.size())
    throw ExcMessage("isotropic boundary inputs are not of the same size");
  for (unsigned int i = 0; i < isotropic_boundary_ids.size(); ++i)
  {
    if (isotropic_bcs.find(isotropic_boundary_ids[i]) != isotropic_bcs.end())
      throw ExcMessage("Duplicate isotropic boundary ids provided");
    if (perpendicular_bcs.find(isotropic_boundary_ids[i]) != perpendicular_bcs.end())
      throw ExcMessage("Boundary has both a perpendicular and an isotropic boundary condition");
    isotropic_bcs.emplace(isotropic_boundary_ids[i], isotropic_boundary_fluxes[i]);
  }

  // Marker for incident BCs
  if (isotropic_bcs.size() != 0 || perpendicular_bcs.size() != 0)
    incident_bcs = true;
}

} // namespace SNProblem
