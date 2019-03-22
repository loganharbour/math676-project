#include "problem.h"

#include <deal.II/numerics/data_out.h>

namespace RadProblem
{
using namespace dealii;

Problem::Problem()
  : ParameterAcceptor("Problem"),
    sn(*this),
    dsa(*this),
    dof_handler(discretization.get_dof_handler())
{
  // .vtu output filename (default: output); no output if empty
  add_parameter("vtu_filename", vtu_filename);
  // Residual vector output filename (default: empty); no output if empty
  add_parameter("residual_filename", residual_filename);

  // Maximum source iterations (default: 1000)
  add_parameter("max_source_iterations", max_its);
  // Source iteration tolerance (defaut: 1e-12)
  add_parameter("source_iteration_tolerance", source_iteration_tol);
}

void
Problem::run()
{
  setup();
  solve();

  // Ouput scalar_flux in .vtu format if filename is given
  if (vtu_filename.length() != 0)
    output_vtu();
  // Output residuals in a text file if filename is given
  if (residual_filename.length() != 0)
    saveVector(residuals, residual_filename);
}

void
Problem::setup()
{
  // Setup mesh
  discretization.setup();

  // Setup description (needs discretization for bc/material coverage)
  description.setup(discretization);

  // Resize scalar flux variables
  scalar_flux.reinit(dof_handler.n_dofs());
  scalar_flux_old.reinit(dof_handler.n_dofs());

  // Setup the problems
  sn.setup();
  dsa.setup();
}

void
Problem::solve()
{
  for (unsigned int l = 0; l < max_its; ++l)
  {
    std::cout << "Source iteration " << l << std::endl;

    // Solve all directions with SN
    sn.solve_directions();

    // Do not iterate without scattering
    if (!description.has_scattering())
      return;

    // Solve for DSA scalar flux correction
    dsa.solve();

    // Compute L2 norm of (scalar_flux - scalar_flux_old) and store
    const double norm = scalar_flux_L2();
    residuals.emplace_back(norm);
    std::cout << "  Scalar flux L2 difference: " << std::scientific << std::setprecision(2) << norm
              << std::endl
              << std::endl;

    // Exit if converged
    if (norm < source_iteration_tol)
      return;
  }

  std::cout << "Did not converge after " << max_its << " source iterations!" << std::endl;
}

double
Problem::scalar_flux_L2() const
{
  double value = 0;

  const auto & fe = dof_handler.get_fe();
  QGauss<2> quadrature(fe.degree + 1);
  FEValues<2> fe_v(fe, quadrature, update_values | update_JxW_values);

  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  for (const auto & cell : dof_handler.active_cell_iterators())
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      const double diff = scalar_flux[indices[i]] - scalar_flux_old[indices[i]];
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        value += std::pow(fe_v.shape_value(i, q) * diff, 2) * fe_v.JxW(q);
    }
  }

  return value;
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
