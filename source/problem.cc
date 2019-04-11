#include "problem.h"

#include <deal.II/numerics/data_out.h>

namespace RadProblem
{
using namespace dealii;

template <int dim>
Problem<dim>::Problem()
  : ParameterAcceptor("Problem"),
    comm(MPI_COMM_WORLD),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0)),
    discretization(comm),
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

template <int dim>
void
Problem<dim>::run()
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

template <int dim>
void
Problem<dim>::setup()
{
  // Setup mesh
  discretization.setup();

  // Setup description (needs discretization for bc/material coverage)
  description.setup(discretization);

  // Resize system variable
  system_rhs.reinit(discretization.get_locally_owned_dofs(), comm);
  system_matrix.reinit(discretization.get_locally_owned_dofs(),
                       discretization.get_locally_owned_dofs(),
                       discretization.get_sparsity_pattern(),
                       comm);
  system_solution.reinit(discretization.get_locally_owned_dofs(), comm);

  // Resize scalar flux variables
  scalar_flux.reinit(discretization.get_locally_owned_dofs(), comm);
  scalar_flux_old.reinit(discretization.get_locally_owned_dofs(), comm);

  // Setup the problems
  sn.setup();
  dsa.setup();
}

template <int dim>
void
Problem<dim>::solve()
{
  for (unsigned int l = 0; l < max_its; ++l)
  {
    pcout << "Source iteration " << l << std::endl;

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
    pcout << "  Scalar flux L2 difference: " << std::scientific << std::setprecision(2) << norm
          << std::endl
          << std::endl;

    // Exit if converged
    if (norm < source_iteration_tol)
      return;
  }

  pcout << "Did not converge after " << max_its << " source iterations!" << std::endl;
}

template <int dim>
double
Problem<dim>::scalar_flux_L2() const
{
  double value = 0;

  const auto & fe = dof_handler.get_fe();
  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_v(fe, quadrature, update_values | update_JxW_values);

  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  for (const auto & cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

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

template <int dim>
void
Problem<dim>::output_vtu() const
{
  const auto & triangulation = discretization.get_triangulation();

  // DataOut setup
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(scalar_flux, "scalar_flux");

  // Build processor partitioning
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();

  // Write output for this local processor
  const std::string filename =
      (vtu_filename + "-" + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  // Write master output
  if (Utilities::MPI::this_mpi_process(comm) == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(comm); ++i)
      filenames.push_back(vtu_filename + "-" + Utilities::int_to_string(i, 4) + ".vtu");
    std::ofstream master_output((vtu_filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template Problem<2>::Problem();
template Problem<3>::Problem();

template void Problem<2>::run();
template void Problem<3>::run();

} // namespace RadProblem
