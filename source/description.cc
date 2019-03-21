#include "description.h"
#include "material.h"

#include <deal.II/base/exceptions.h>

#include <algorithm>

namespace RadProblem
{
using namespace dealii;

Description::Description() : ParameterAcceptor("Description")
{
  // Add material parameters (default: empty)
  add_parameter("material_ids", material_ids);
  add_parameter("material_sigma_t", material_sigma_t);
  add_parameter("material_sigma_s", material_sigma_s);
  add_parameter("material_src", material_src);

  // Add boundary parameters
  add_parameter("perpendicular_boundary_ids", perpendicular_boundary_ids);
  add_parameter("perpendicular_boundary_fluxes", perpendicular_boundary_fluxes);
  add_parameter("isotropic_boundary_ids", isotropic_boundary_ids);
  add_parameter("isotropic_boundary_fluxes", isotropic_boundary_fluxes);
  add_parameter("reflective_boundary_ids", reflective_boundary_ids);
  add_parameter("vacuum_boundary_ids", vacuum_boundary_ids); // default: {0}
}

void
Description::setup(const std::set<unsigned int> & mesh_material_ids)
{
  setup_bcs();
  setup_materials(mesh_material_ids);
}

void
Description::setup_materials(const std::set<unsigned int> & mesh_material_ids)
{
  // Make sure all material vectors are the same size
  const auto & double_inputs = {material_sigma_t, material_sigma_s, material_src};
  for (const auto & input : double_inputs)
    if (input.size() != material_ids.size())
      throw ExcMessage("Material input size mismatch");

  // Make sure we have a user-defined material for each mesh material id
  for (auto it = mesh_material_ids.begin(); it != mesh_material_ids.end(); ++it)
    if (std::find(material_ids.begin(), material_ids.end(), *it) == material_ids.end())
      throw ExcMessage("Material missing for id " + std::to_string(*it));

  // Make sure user isn't providing extra materials that don't exist in the mesh
  if (material_ids.size() != mesh_material_ids.size())
    throw ExcMessage("Extraneous materials provided");

  for (unsigned int i = 0; i < material_ids.size(); ++i)
  {
    // Marker for if the problem has scattering
    if (material_sigma_s[i] > 0)
      scattering = true;
    // Check for duplicate ids
    if (materials.find(material_ids[i]) != materials.end())
      throw ExcMessage("Material id " + std::to_string(material_ids[i]) +
                       " provided more than once");
    // Insert into material map
    materials.emplace(material_ids[i],
                      Material(material_sigma_t[i], material_sigma_s[i], material_src[i]));
  }
}

void
Description::setup_bcs()
{
  fill_bcs(BCTypes::Perpendicular, perpendicular_boundary_ids, &perpendicular_boundary_fluxes);
  fill_bcs(BCTypes::Isotropic, isotropic_boundary_ids, &isotropic_boundary_fluxes);
  fill_bcs(BCTypes::Reflective, reflective_boundary_ids);
  fill_bcs(BCTypes::Vacuum, vacuum_boundary_ids);
}

void
Description::fill_bcs(const BCTypes type,
                      const std::vector<unsigned int> & ids,
                      const std::vector<double> * values)
{
  // Check for uniqueness
  for (unsigned int id : ids)
  {
    if (input_bc_ids.find(id) != input_bc_ids.end())
      throw ExcMessage("Boundary IDs are not unique");
    else
      input_bc_ids.insert(id);
  }

  // Check for matching size (if values are given)
  if (values && ids.size() != values->size())
    throw ExcMessage("Boundary inputs are not of the same size");

  // Fill with values if given
  if (values)
    for (unsigned int i = 0; i < ids.size(); ++i)
      bcs.emplace(ids[i], BC(type, (*values)[i]));
  else
    for (unsigned int i = 0; i < ids.size(); ++i)
      bcs.emplace(ids[i], BC(type));

  // Set identifier for incident BCs if necessary
  if (ids.size() != 0 && type != BCTypes::Vacuum)
    incident_bcs = true;
}

const Description::BC &
Description::get_bc(const unsigned int boundary_id) const
{
  const auto search = bcs.find(boundary_id);
  Assert(search != bcs.end(), ExcMessage("Boundary id not found in BC map"));
  return search->second;
}

const Description::Material &
Description::get_material(const unsigned int material_id) const
{
  const auto search = materials.find(material_id);
  Assert(search != materials.end(), ExcMessage("Material id not found in material map"));
  return search->second;
}

} // namespace RadProblem
