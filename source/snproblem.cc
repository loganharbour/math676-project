#include "snproblem.h"

#include "angular_quadrature.h"
#include "description.h"
#include "discretization.h"
#include "problem.h"

namespace RadProblem
{
using namespace dealii;

template <int dim>
SNProblem<dim>::SNProblem(Problem<dim> & problem)
  : ParameterAcceptor("SNProblem"),
    comm(problem.get_comm()),
    pcout(problem.get_pcout()),
    timer(problem.get_timer()),
    description(problem.get_description()),
    discretization(problem.get_discretization()),
    dof_handler(discretization.get_dof_handler()),
    aq(discretization.get_aq()),
    scalar_flux(problem.get_scalar_flux()),
    scalar_flux_old(problem.get_scalar_flux_old()),
    angular_flux(problem.get_angular_flux()),
    reflective_dof_normals(problem.get_reflective_dof_normals()),
    reflective_incoming_flux(problem.get_reflective_incoming_flux()),
    reflective_dJ(problem.get_reflective_dJ()),
    system_matrix(problem.get_system_matrix()),
    system_rhs(problem.get_system_rhs()),
    system_solution(problem.get_system_solution())
{
  // Whether or not to enable detailed solver output (default: false)
  add_parameter("detailed_solver_output", detailed_solver_output);
  // Relative tolerance (default: 1e-12)
  add_parameter("relative_tolerance", relative_tolerance);
  // Absolute tolerance (default: 1e-12)
  add_parameter("absolute_tolerance", absolute_tolerance);
}

template <int dim>
void
SNProblem<dim>::setup()
{
  TimerOutput::Scope t(timer, "SNProblem setup");

  // Setup InfoBox for MeshWorker
  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_JxW_values | update_values | update_gradients;
  info_box.add_update_flags_all(update_flags);
  info_box.initialize(dof_handler.get_fe(), discretization.get_mapping());

  // Pass the matrix and rhs to the assembler
  assembler.initialize(system_matrix, system_rhs);
}

template <int dim>
void
SNProblem<dim>::assemble_solve_update()
{
  for (unsigned int d = 0; d < aq.n_dir(); ++d)
    assemble_solve_update(d);
}

template <int dim>
void
SNProblem<dim>::assemble_solve_update(const unsigned int d)
{
  // Updates for reflecting bcs before sweep (incoming current on reflective boundaries)
  if (description.has_reflecting_bcs())
    update_for_reflective_bc(d, true);

  // Assemble the system
  assemble(d);

  // And solve
  solve(d);

  // Updates for reflecting bcs after sweep (outgoing current on reflective boundaries
  // and update incoming angular fluxes)
  if (description.has_reflecting_bcs())
    update_for_reflective_bc(d, false);

  // Update angular flux if enabled
  if (angular_flux.size() != 0)
    angular_flux[d] = system_solution;

  // Update scalar flux at each node
  system_solution *= aq.w(d);
  scalar_flux += system_solution;
}

template <int dim>
void
SNProblem<dim>::solve(const unsigned int d)
{
  TimerOutput::Scope t(timer, "SNProblem solve");

  system_solution = 0;
  SolverControl control(1000, relative_tolerance * system_rhs.l2_norm() + absolute_tolerance);
  TrilinosWrappers::SolverGMRES::AdditionalData gmres_data(detailed_solver_output);
  TrilinosWrappers::SolverGMRES solver(control, gmres_data);
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix);
  solver.solve(system_matrix, system_solution, system_rhs, preconditioner);

  pcout << "  Direction " << d << " converged after " << control.last_step() << " GMRES iterations "
        << std::endl;
}

template <int dim>
void
SNProblem<dim>::assemble(const unsigned int d)
{
  TimerOutput::Scope t(timer, "SNProblem assembly");

  // Zero lhs and rhs before assembly
  system_matrix = 0;
  system_rhs = 0;

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info) {
    SNProblem::integrate_cell(dinfo, info, d);
  };
  const auto boundary_worker = [&](MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info) {
    SNProblem::integrate_boundary(dinfo, info, d);
  };
  const auto face_worker = [&](MeshWorker::DoFInfo<dim> & dinfo1,
                               MeshWorker::DoFInfo<dim> & dinfo2,
                               MeshWorker::IntegrationInfo<dim> & info1,
                               MeshWorker::IntegrationInfo<dim> & info2) {
    SNProblem::integrate_face(dinfo1, dinfo2, info1, info2, d);
  };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<dim> dof_info(dof_handler);
  MeshWorker::LoopControl loop_control;
  // With faces_to_ghost = both,
  loop_control.faces_to_ghost = MeshWorker::LoopControl::FaceOption::both;
  MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      cell_worker,
      boundary_worker,
      face_worker,
      assembler,
      loop_control);
}

template <int dim>
void
SNProblem<dim>::integrate_cell(MeshWorker::DoFInfo<dim> & dinfo,
                               MeshWorker::IntegrationInfo<dim> & info,
                               const unsigned int d) const
{
  const auto dir = aq.dir(d);
  const FEValuesBase<dim> & fe_v = info.fe_values();
  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Material for this cell
  const auto & material = description.get_material(dinfo.cell->material_id());

  // Whether or not this cell has scattering
  const bool has_scattering = material.sigma_s != 0;

  // Compute old scalar flux at each quadrature point for scattering source
  std::vector<double> scattering_source;
  if (has_scattering)
  {
    double scalar_flux_old_i;
    scattering_source.resize(fe_v.n_quadrature_points);
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      scalar_flux_old_i = scalar_flux_old[dinfo.indices[i]];
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        scattering_source[q] += material.sigma_s * scalar_flux_old_i * fe_v.shape_value(i, q);
    }
  }

  for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      const double v_i = fe_v.shape_value(i, q);
      const double dir_dot_grad_v_i = dir * fe_v.shape_grad(i, q);
      for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
        // Streaming + collision
        local_matrix(i, j) +=
            (fe_v.shape_value(j, q) * JxW[q]) * (material.sigma_t * v_i - dir_dot_grad_v_i);
      // External source
      local_vector(i) += v_i * material.src * JxW[q];
      // Scattering source
      if (has_scattering)
        local_vector(i) += v_i * scattering_source[q] * JxW[q];
    }
}

template <int dim>
void
SNProblem<dim>::integrate_boundary(MeshWorker::DoFInfo<dim> & dinfo,
                                   MeshWorker::IntegrationInfo<dim> & info,
                                   const unsigned int d) const
{
  const auto & fe = dof_handler.get_fe();
  const FEValuesBase<dim> & fe_v = info.fe_values();

  // Dot product between the direction and the outgoing normal
  const auto normal = info.fe_values().normal_vector(0);
  const double dir_dot_n = aq.dir(d) * normal;

  // Face is incoming
  if (dir_dot_n < 0)
  {
    // Nothing to do for incoming without incident boundary conditions
    if (!description.has_incident_bcs())
      return;

    // Set the values at each qp as necessary depending on type
    std::vector<double> bc_values(fe_v.n_quadrature_points, 0);
    const auto & bc = description.get_bc(dinfo.face->boundary_id());
    // Isotropic boundary conditions or incident into this direction
    if (bc.type == BCTypes::Isotropic || (bc.type == BCTypes::Incident && bc.d == d))
    {
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        bc_values[q] = bc.value;
    }
    // Reflective boundary condition
    else if (bc.type == BCTypes::Reflective)
    {
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
      {
        // Dof is not on this face
        if (!fe.has_support_on_face(i, dinfo.face_number))
          continue;
        // Incoming reflected angular flux at the i-th local dof
        const double flux_i = reflective_incoming_flux[d].at(dinfo.indices[i]);
        // Accumulate at quadrature points
        for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
          bc_values[q] += fe_v.shape_value(i, q) * flux_i;
      }
    }
    // The other boundary types do not contribute to incoming faces
    else
      return;

    Vector<double> & local_vector = dinfo.vector(0).block(0);
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        if (fe.has_support_on_face(i, dinfo.face_number))
          local_vector(i) += (-dir_dot_n) * bc_values[q] * fe_v.shape_value(i, q) * JxW[q];
  }
  // Face is outgoing
  else
  {
    FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        if (fe.has_support_on_face(i, dinfo.face_number))
        {
          const double coeff = dir_dot_n * fe_v.shape_value(i, q) * JxW[q];
          for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
            local_matrix(i, j) += coeff * fe_v.shape_value(j, q);
        }
  }
}

template <int dim>
void
SNProblem<dim>::integrate_face(MeshWorker::DoFInfo<dim> & dinfo1,
                               MeshWorker::DoFInfo<dim> & dinfo2,
                               MeshWorker::IntegrationInfo<dim> & info1,
                               MeshWorker::IntegrationInfo<dim> & info2,
                               const unsigned int d) const
{
  // Dot product between the direction and the outgoing normal of face 1
  const double dir_dot_n1 = aq.dir(d) * info1.fe_values().normal_vector(0);

  // Whether the first cell is the outgoing cell or not
  const bool cell1_out = dir_dot_n1 > 0;

  // FE values and dinfo access for the outgoing and incoming cell
  const auto & fe_v_out = (cell1_out ? info1.fe_values() : info2.fe_values());
  const auto & fe_v_in = (cell1_out ? info2.fe_values() : info1.fe_values());
  auto & dinfo_out = (cell1_out ? dinfo1 : dinfo2);
  auto & dinfo_in = (cell1_out ? dinfo2 : dinfo1);

  // System matrices for the u_out v_out matrix and the u_out v_in matrix
  auto & uout_vout_matrix = dinfo_out.matrix(0, false).matrix;
  auto & uout_vin_matrix = dinfo_in.matrix(0, true).matrix;

  // Reverse the direction for the dot product if cell 2 is the outgoing cell
  const double dir_dot_nout = (cell1_out ? dir_dot_n1 : -dir_dot_n1);

  // Use quadrature points from cell 1; reference element is the same
  const auto & fe = dof_handler.get_fe();
  const std::vector<double> & JxW = info1.fe_values().get_JxW_values();
  for (unsigned int q = 0; q < fe_v_out.n_quadrature_points; ++q)
    for (unsigned int j = 0; j < fe_v_out.dofs_per_cell; ++j)
    {
      if (!fe.has_support_on_face(j, dinfo_out.face_number))
        continue;
      const double coeff = dir_dot_nout * fe_v_out.shape_value(j, q) * JxW[q];
      for (unsigned int i = 0; i < fe_v_out.dofs_per_cell; ++i)
        if (fe.has_support_on_face(i, dinfo_out.face_number))
          uout_vout_matrix(i, j) += coeff * fe_v_out.shape_value(i, q);
      for (unsigned int i = 0; i < fe_v_in.dofs_per_cell; ++i)
        if (fe.has_support_on_face(i, dinfo_in.face_number))
          uout_vin_matrix(i, j) -= coeff * fe_v_in.shape_value(i, q);
    }
}

template <int dim>
void
SNProblem<dim>::update_for_reflective_bc(const unsigned int d, const bool before_sweep)
{
  TimerOutput::Scope t(timer, "SNProblem reflective BC update");

  // Loop over each reflective dof and its corresponding outward normal
  for (const auto & dof_normal_pair : reflective_dof_normals)
  {
    const unsigned int dof = dof_normal_pair.first;
    const HatDirection normal_hat = dof_normal_pair.second;
    const Tensor<1, dim> normal = get_hat_direction<dim>(normal_hat);
    const double dir_dot_n = aq.dir(d) * normal;

    // Before sweep, incoming directions only
    if (before_sweep && dir_dot_n < 0)
      // Update incoming current
      reflective_dJ[dof] += aq.w(d) * dir_dot_n * reflective_incoming_flux[d].at(dof);
    // After sweep, outgoing directions only
    else if (!before_sweep && dir_dot_n > 0)
    {
      // Update outgoing current
      reflective_dJ[dof] += aq.w(d) * dir_dot_n * system_solution[dof];
      // Update incoming angular flux on dofs that this direction reflects into
      const auto d_to = aq.reflect_to(normal_hat, d);
      reflective_incoming_flux[d_to][dof] = system_solution[dof];
    }
  }
}

template SNProblem<2>::SNProblem(Problem<2> & problem);
template SNProblem<3>::SNProblem(Problem<3> & problem);

template void SNProblem<2>::setup();
template void SNProblem<3>::setup();

template void SNProblem<2>::assemble_solve_update();
template void SNProblem<3>::assemble_solve_update();

} // namespace RadProblem
