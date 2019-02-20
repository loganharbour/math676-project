#ifndef DESCRIPTION_H
#define DESCRIPTION_H

#include <map>

namespace SNProblem
{
// Forward declaration
class Material;

class Description
{
public:
  Description();

  void add_material(const unsigned int id, const Material & material);
  const std::map<const unsigned int, const Material> & get_materials() const { return materials; }

  // Material properties
  std::map<const unsigned int, const Material> materials;
};
} // namespace SNProblem

#endif // DESCRIPTION_H
