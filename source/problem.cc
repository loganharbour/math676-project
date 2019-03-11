#include "problem.h"

namespace SNProblem
{
using namespace dealii;

Problem::Problem()
  : dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
    aq(discretization.get_aq())
{
}

void
Problem::setup()
{
  // Setup other classes first
  description.setup();
  discretization.setup();

  // Initialize system storage for a single direction
  direction_rhs.reinit(dof_handler.n_dofs());
  direction_matrix.reinit(discretization.get_sparsity_pattern());

  // Resize system variables
  direction_solution.reinit(dof_handler.n_dofs());
  scalar_flux.reinit(dof_handler.n_dofs());
  scalar_flux_old.reinit(dof_handler.n_dofs());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(direction_matrix, direction_rhs);
}

void
Problem::solve()
{
  for (unsigned int l = 0; l < 1000; ++l)
  {
    // Zero scalar flux for update
    scalar_flux = 0;

    // Solve each direction
    for (unsigned int d = 0; d < aq.n_dir(); ++d)
      solve_direction(d);

    // Source iteration: check for convergence
    if (description.has_scattering())
    {
      const double norm = L2_difference(scalar_flux, scalar_flux_old);
      std::cout << "Source iteration " << l << " norm: " << std::scientific << std::setprecision(2)
                << norm << std::endl
                << std::endl;

      // If converged, exit
      if (norm < 1e-12)
        return;
      // If not converged, copy to old scalar flux
      else
        scalar_flux_old = scalar_flux;
    }
    // No scattering: no source iteration required
    else
      return;
  }

  std::cout << "Source iteration did not converge" << std::endl;
}

void
Problem::run()
{
  setup();
  solve();
}

} // namespace SNProblem
