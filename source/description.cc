#include "description.h"

#include "discretization.h"

#include <deal.II/base/exceptions.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
Description<dim>::Description() : ParameterAcceptor("Description")
{
  // Add material parameters (default: empty)
  add_parameter("material_ids", material_ids);
  add_parameter("material_sigma_t", material_sigma_t);
  add_parameter("material_sigma_s", material_sigma_s);
  add_parameter("material_src", material_src);

  // Add boundary parameters
  add_parameter("isotropic_boundary_ids", isotropic_boundary_ids);
  add_parameter("isotropic_boundary_fluxes", isotropic_boundary_fluxes);
  add_parameter("reflective_boundary_ids", reflective_boundary_ids);
  add_parameter("vacuum_boundary_ids", vacuum_boundary_ids);
}

template <int dim>
void
Description<dim>::setup(const Discretization<dim> & discretization)
{
  setup_bcs(discretization.get_boundary_ids());
  setup_materials(discretization.get_material_ids());
}

template <int dim>
void
Description<dim>::setup_materials(const std::set<unsigned int> & mesh_material_ids)
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
  // or one material more than once
  if (material_ids.size() != mesh_material_ids.size())
    throw ExcMessage("Extraneous materials provided");

  for (unsigned int i = 0; i < material_ids.size(); ++i)
  {
    // Marker for if the problem has scattering
    if (material_sigma_s[i] > 0)
      scattering = true;
    // Insert into material map
    materials.emplace(material_ids[i],
                      Material(material_sigma_t[i], material_sigma_s[i], material_src[i]));
  }
}

template <int dim>
void
Description<dim>::setup_bcs(const std::set<unsigned int> & mesh_boundary_ids)
{
  fill_bcs(BCTypes::Isotropic, isotropic_boundary_ids, &isotropic_boundary_fluxes);
  fill_bcs(BCTypes::Reflective, reflective_boundary_ids);
  fill_bcs(BCTypes::Vacuum, vacuum_boundary_ids);

  // Check for a BC for every boundary id
  for (auto it = mesh_boundary_ids.begin(); it != mesh_boundary_ids.end(); ++it)
    if (bcs.find(*it) == bcs.end())
      throw ExcMessage("Missing boundary condition for boundary id " + std::to_string(*it));
}

template <int dim>
void
Description<dim>::fill_bcs(const BCTypes type,
                           const std::vector<unsigned int> & ids,
                           const std::vector<double> * values)
{
  // Check for matching size (if values are given)
  if (values && ids.size() != values->size())
    throw ExcMessage("Boundary input values are not of the same size");

  for (unsigned int i = 0; i < ids.size(); ++i)
  {
    // Check for uniqueness
    if (bcs.find(ids[i]) != bcs.end())
      throw ExcMessage("Boundary id " + std::to_string(ids[i]) + " given more than once");
    // Fill a BC that includes a value
    if (values)
      bcs.emplace(ids[i], BC(type, (*values)[i]));
    // Fill a BC that does not include a value
    else
      bcs.emplace(ids[i], BC(type));
  }

  if (ids.size() != 0)
  {
    if (type != BCTypes::Vacuum)
      incident_bcs = true;
    if (type == BCTypes::Reflective)
      reflecting_bcs = true;
  }
}

template <int dim>
const BC &
Description<dim>::get_bc(const unsigned int boundary_id) const
{
  const auto search = bcs.find(boundary_id);
  Assert(search != bcs.end(), ExcMessage("Boundary id not found in BC map"));
  return search->second;
}

template <int dim>
const Material &
Description<dim>::get_material(const unsigned int material_id) const
{
  const auto search = materials.find(material_id);
  Assert(search != materials.end(), ExcMessage("Material id not found in material map"));
  return search->second;
}

template Description<2>::Description();
template Description<3>::Description();

template void Description<2>::setup(const Discretization<2> & discretization);
template void Description<3>::setup(const Discretization<3> & discretization);

template const BC & Description<2>::get_bc(const unsigned int boundary_id) const;
template const BC & Description<3>::get_bc(const unsigned int boundary_id) const;

template const Material & Description<2>::get_material(const unsigned int material_id) const;
template const Material & Description<3>::get_material(const unsigned int material_id) const;

} // namespace RadProblem
