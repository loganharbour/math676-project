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
    timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    discretization(comm, timer),
    aq(discretization.get_aq()),
    sn(*this),
    dsa(*this),
    dof_handler(discretization.get_dof_handler())
{
  // .vtu output filename (default: output); no output if empty
  add_parameter("vtu_filename", vtu_filename);
  // Residual vector output filename (default: empty); no output if empty
  add_parameter("residual_filename", residual_filename);
  // Whether or not to save the angular flux (default: false)
  add_parameter("save_angular_flux", save_angular_flux);

  // Maximum source iterations (default: 1000)
  add_parameter("max_source_iterations", max_source_its);
  // Source iteration tolerance (defaut: 1e-12)
  add_parameter("source_iteration_tolerance", source_iteration_tol);

  // Maximum source iterations (default: 1)
  add_parameter("max_reflective_iterations", max_ref_its);
  // Source iteration tolerance (defaut: 1e-16)
  add_parameter("reflective_iteration_tolerance", reflective_iteration_tol);

  // Enable DSA (default: true)
  add_parameter("dsa", enable_dsa);
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

  TimerOutput::Scope t(timer, "Problem setup");

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

  // Resize angular flux variables
  if (save_angular_flux)
  {
    angular_flux.resize(aq.n_dir());
    for (unsigned int d = 0; d < aq.n_dir(); ++d)
      angular_flux[d].reinit(discretization.get_locally_owned_dofs(), comm);
  }

  // Sanity checks on reflecting boundary condition iteraions
  if (!description.has_scattering() && description.has_reflecting_bcs() && max_ref_its == 1)
  {
    pcout << "\nProblem max_ref_its is set to 1, but there is no scattering and therefore\n"
          << " the reflective boundary conditions will likely not converge. max_ref_its is\n"
          << " being set to 4. Set it greater than 1 to escape this warning.\n\n";
    max_ref_its = 4;
  }
  if (description.has_scattering() && !enable_dsa && max_ref_its == 1)
  {
    pcout << "\nProblem max_ref_its is set to 1 because DSA can converge the reflecting\n"
          << " boundary conditions. However, you have disabled DSA and therefore the\n"
          << " reflecting boundary conditions will likely not converge. max_ref_its is\n"
          << " being set to 4. Set it greater than 1 to escape this warning.\n\n";
    max_ref_its = 4;
  }
  if (max_ref_its == 0)
    throw ExcMessage("max_ref_its in Problem must be > 0");

  // Allocate storage for reflecting boundaries
  if (description.has_reflecting_bcs())
  {
    reflective_incoming_flux.resize(aq.n_dir());

    QGauss<dim - 1> quadrature(1);
    FEFaceValues<dim> fe(dof_handler.get_fe(), quadrature, update_normal_vectors);
    std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
    std::vector<types::global_dof_index> face_dofs(GeometryInfo<dim>::vertices_per_face);
    for (const auto & cell : dof_handler.active_cell_iterators())
    {
      // Skip non-local and non-boundary cells
      if (!cell->is_locally_owned() || !cell->at_boundary())
        continue;

      // Check each face
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        // Skip non-boundary faces and non-reflective faces
        const auto & face = cell->face(f);
        if (!face->at_boundary() ||
            description.get_bc(face->boundary_id()).type != BCTypes::Reflective)
          continue;

        // Outward-facing unit normal for this face
        fe.reinit(cell, f);
        const Tensor<1, dim> normal = fe.normal_vector(0);
        const HatDirection normal_hat = get_hat_direction<dim>(normal);

        // Dofs for this face
        cell->get_dof_indices(dofs);
        for (unsigned int fv = 0; fv < face_dofs.size(); ++fv)
          face_dofs[fv] = dofs[GeometryInfo<dim>::face_to_cell_vertices(f, fv)];

        for (const types::global_dof_index dof : face_dofs)
        {
          // Initialize storage for the net current on reflective boundaries
          reflective_dJ.emplace(dof, 0);
          // Store this normal for computation of the reflected flux integral later
          reflective_dof_normals.emplace(dof, normal_hat);
          // Initialize storage for angular fluxes on incoming reflective boundaries
          for (unsigned int d = 0; d < aq.n_dir(); ++d)
            if (aq.dir(d) * normal < 0)
              reflective_incoming_flux[d].emplace(dof, 0);
        } // dofs on a face
      }   // faces
    }     // cells
  }       // has reflecting bcs

  // Setup the problems
  sn.setup();
  if (enable_dsa && description.has_scattering())
    dsa.setup();
}

template <int dim>
void
Problem<dim>::solve()
{
  // Norm of the current on reflecting boundaries
  double reflective_dJ_norm;

  for (unsigned int l_source = 0; l_source < max_source_its; ++l_source)
  {
    pcout << "Source iteration " << l_source << std::endl;

    // Copy to old scalar flux (only needed between source iterations)
    scalar_flux_old = scalar_flux;

    for (unsigned int l_ref = 0; l_ref < max_ref_its; ++l_ref)
    {
      // Zero for scalar flux update
      scalar_flux = 0;

      if (description.has_reflecting_bcs())
      {
        pcout << "  Reflective iteration " << l_ref << std::endl;
        // Zero dJ on reflective boundaries
        for (auto & dof_value_pair : reflective_dJ)
          dof_value_pair.second = 0;
      }

      // Assmemble, solve, and update all directions with SN
      sn.assemble_solve_update();

      // Check for convergence of the reflecting boundaries
      if (description.has_reflecting_bcs())
      {
        // Compute norm of dJ on the reflective boundary
        reflective_dJ_norm = reflective_dJ_L2();
        pcout << "  Reflecting BC net current L2 norm: " << std::scientific << std::setprecision(2)
              << reflective_dJ_norm << std::endl
              << std::endl;
        // Converged: exit reflecting boundary loop
        if (reflective_dJ_norm < reflective_iteration_tol)
          break;
      }
      // No reflecting boundaries: exit reflecting boundary loop
      else
        break;
    } // End reflecting boundary iterations

    // No source iteration without scattering
    if (!description.has_scattering())
      return;

    // Assemble, solve, and update with DSA
    if (enable_dsa)
      dsa.assemble_solve_update();

    // Compute L2 norm of (scalar_flux - scalar_flux_old) and store
    residuals.emplace_back(scalar_flux_L2());
    pcout << "  Scalar flux L2 difference: " << std::scientific << std::setprecision(2)
          << residuals.back() << std::endl
          << std::endl;

    // If we have reflecting boundaries and they are still not converged, continue
    // to the next source iteration (don't check for source iteration convergence)
    if (description.has_reflecting_bcs() && reflective_dJ_norm > reflective_iteration_tol)
    {
      pcout << "\nSource iteration has technically converged but the reflective\n"
            << " iterations have not! Consider increasing max_ref_its!\n\n";
      continue;
    }

    // Break source iterations if converged
    if (residuals.back() < source_iteration_tol)
      return;

  } // End source iterations

  pcout << "Did not converge after " << max_source_its << " source iterations!" << std::endl;
}

template <int dim>
double
Problem<dim>::scalar_flux_L2()
{
  TimerOutput::Scope t(timer, "Problem scalar flux L2 norm");

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

  // Gather among all processors
  return Utilities::MPI::sum(value, comm);
}

template <int dim>
double
Problem<dim>::reflective_dJ_L2()
{
  TimerOutput::Scope t(timer, "Problem reflective dJ L2 norm");

  const auto & fe = dof_handler.get_fe();
  QGauss<dim - 1> quadrature(fe.degree + 1);
  FEFaceValues<dim> fe_v(fe, quadrature, update_values | update_JxW_values);
  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  double value = 0;

  for (const auto & cell : dof_handler.active_cell_iterators())
  {
    // Immediately skip non-local and non-boundary cells
    if (!cell->is_locally_owned() || !cell->at_boundary())
      continue;

    cell->get_dof_indices(indices);

    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      // Skip non-boundary faces and non-reflective faces
      const auto & face = cell->face(f);
      if (!face->at_boundary() ||
          description.get_bc(face->boundary_id()).type != BCTypes::Reflective)
        continue;

      fe_v.reinit(cell, f);
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        if (fe.has_support_on_face(i, f))
        {
          const double dJ = reflective_dJ.at(indices[i]);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            value += std::pow(fe_v.shape_value(i, q) * dJ, 2) * fe_v.JxW(q);
        }
    }
  }

  // Gather among all processors
  return Utilities::MPI::sum(value, comm);
}

template <int dim>
void
Problem<dim>::output_vtu()
{
  TimerOutput::Scope t(timer, "Problem output");

  const auto & triangulation = discretization.get_triangulation();

  // DataOut setup
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(scalar_flux, "scalar_flux");
  if (save_angular_flux)
    for (unsigned int d = 0; d < aq.n_dir(); ++d)
      data_out.add_data_vector(angular_flux[d], "angular_flux_d" + Utilities::int_to_string(d));

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
