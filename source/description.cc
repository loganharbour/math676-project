#include "description.h"
#include "material.h"

#include <deal.II/base/exceptions.h>

namespace SNProblem
{
using namespace dealii;

Description::Description() : ParameterAcceptor("Description")
{
  input_material_id = {0};
  add_parameter("material_id", input_material_id);

  input_material_sigma_t = {100.0};
  add_parameter("material_sigma_t", input_material_sigma_t);

  input_material_sigma_s = {0.0};
  add_parameter("material_sigma_s", input_material_sigma_s);

  input_material_src = {1.0};
  add_parameter("material_src", input_material_src);
}

void
Description::setup()
{
  setup_materials();
}

void
Description::setup_materials()
{
  // Make sure all material vectors are the same size
  const auto & double_inputs = {input_material_sigma_t, input_material_sigma_s, input_material_src};
  for (const auto input : double_inputs)
    if (input.size() != input_material_id.size())
      throw ExcMessage("Material input size mismatch");

  // Create a Material for each id
  for (unsigned int i = 0; i < input_material_id.size(); ++i)
  {
    if (materials.find(input_material_id[i]) != materials.end())
      throw ExcMessage("Material id already exists in material map");
    materials.emplace(input_material_id[i],
                      Material(input_material_sigma_t[i],
                               input_material_sigma_s[i],
                               input_material_src[i]));
  }
}
} // namespace SNProblem
