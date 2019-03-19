#include "dsaproblem.h"

#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>

namespace RadProblem
{
using namespace dealii;

DSAProblem::DSAProblem(Problem & problem)
  : ParameterAcceptor("DSAProblem"),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    materials(description.get_materials()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old())
{
  // Whether or not DSA is enabled (default: true)
  add_parameter("enabled", enabled);
}

void
DSAProblem::setup()
{
  // Do not setup without scattering or if it is disabled
  if (!description.has_scattering() || !enabled)
    return;

  // Initialize system storage for a single direction
  rhs.reinit(dof_handler.n_dofs());
  matrix.reinit(discretization.get_sparsity_pattern());
  solution.reinit(dof_handler.n_dofs());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(matrix, rhs);
}

} // namespace RadProblem
