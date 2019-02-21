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

  std::map<const unsigned int, const Material> materials;

  std::vector<unsigned int> input_material_id;
  std::vector<double> input_material_sigma_t;
  std::vector<double> input_material_sigma_s;
  std::vector<double> input_material_src;
};
} // namespace SNProblem

#endif // DESCRIPTION_H
