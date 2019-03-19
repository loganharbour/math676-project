#include "problem.h"

namespace RadProblem
{
using namespace dealii;

Problem::Problem()
  : ParameterAcceptor("Problem"),
    dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
    aq(discretization.get_aq()),
    sn(*this)
{
  // .vtu output filename (default: output); no output if empty
  add_parameter("vtu_filename", vtu_filename);
}

void
Problem::run()
{
  setup();
  solve();

  if (vtu_filename.length() != 0)
    output_vtu();
}

void
Problem::setup()
{
  // Setup mesh
  discretization.setup();

  // Setup description (which requres material ids owned by the mesh)
  std::set<unsigned int> mesh_material_ids;
  discretization.get_material_ids(mesh_material_ids);
  description.setup(mesh_material_ids);

  // Setup SN problem
  sn.setup();

  // Resize scalar flux variables
  scalar_flux.reinit(dof_handler.n_dofs());
  scalar_flux_old.reinit(dof_handler.n_dofs());
}

void
Problem::solve()
{
  for (unsigned int l = 0; l < 1000; ++l) {
    const bool sn_converged = sn.solve_directions(l);
    if (sn_converged)
      return;
  }

  std::cout << "Source iteration did not converge" << std::endl;
}

void
Problem::output_vtu() const
{
  std::ofstream output(vtu_filename + ".vtu");
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(scalar_flux, "scalar_flux");
  data_out.build_patches();
  data_out.write_vtu(output);
}

} // namespace RadProblem
