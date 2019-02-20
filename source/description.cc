#include "description.h"
#include "material.h"

#include <deal.II/base/exceptions.h>

namespace SNProblem
{
using namespace dealii;

Description::Description() {}

void
Description::add_material(const unsigned int id, const Material & material)
{
  if (materials.find(id) != materials.end())
    throw ExcMessage("Material id already exists in material map");
  else
    materials.emplace(id, material);
}
} // namespace SNProblem
