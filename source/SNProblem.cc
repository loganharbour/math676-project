#include "SNProblem.h"

#include <fstream>

using namespace dealii;

SNProblem::SNProblem() : mapping(), fe(1), dof_handler(triangulation) {}

void
SNProblem::setup_system()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(5);

  aq.init(10);

  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  phi.reinit(dof_handler.n_dofs());
  phi_old.reinit(dof_handler.n_dofs());
  rhs.reinit(dof_handler.n_dofs());
  solution.reinit(dof_handler.n_dofs());

  const unsigned int n_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(n_points, n_points, n_points);

  info_box.initialize_update_flags();
  UpdateFlags update_flags = update_quadrature_points | update_values | update_gradients;
  info_box.add_update_flags(update_flags, true, true, true, true);

  info_box.initialize(fe, mapping);
  assembler.initialize(system_matrix, rhs);
}

void
SNProblem::assemble_direction(unsigned int d)
{
  const auto dir = aq.dir(d);
  system_matrix = 0;
  rhs = 0;

  MeshWorker::DoFInfo<2> dof_info(dof_handler);

  // Lambda functions for passing into MeshWorker::loop
  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    SNProblem::integrate_cell(dinfo, info, dir);
  };
  const auto boundary_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    SNProblem::integrate_boundary(dinfo, info, dir);
  };
  const auto face_worker =
      [&](DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2) {
        SNProblem::integrate_face(dinfo1, dinfo2, info1, info2, dir);
      };

  MeshWorker::loop<2, 2, MeshWorker::DoFInfo<2>, MeshWorker::IntegrationInfoBox<2>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      info_box,
      cell_worker,
      boundary_worker,
      face_worker,
      assembler);
}

void
SNProblem::integrate_cell(DoFInfo & dinfo, CellInfo & info, Point<2> dir)
{
  const FEValuesBase<2> & fe_v = info.fe_values();
  const unsigned int n_q = fe_v.n_quadrature_points;
  const unsigned int n_dofs = fe_v.dofs_per_cell;

  FullMatrix<double> & local_matrix = dinfo.matrix(0).matrix;
  Vector<double> & local_vector = dinfo.vector(0).block(0);
  const std::vector<double> & JxW = fe_v.get_JxW_values();

  // Get material for this cell
  const unsigned int material_id = dinfo.cell->material_id();
  Assert(materials.find(material_id) != materials.end(),
         ExcMessage("Material id not found in material map"));
  const Material & material = materials.at(material_id);
  const bool has_scattering = material.sigma_s != 0;

  // Obtain the old scalar flux if scattering exists in this cell
  std::vector<double> local_phi_old;
  if (has_scattering) {
    local_phi_old.resize(n_dofs);
    for (unsigned int i = 0; i < n_dofs; ++i)
      local_phi_old[i] = phi[dinfo.indices[i]];
  }

  for (unsigned int q = 0; q < n_q; ++q)
  {
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const double u_i = fe_v.shape_value(i, q);
      double phi_old_q = 0;
      for (unsigned int j = 0; j < n_dofs; ++j)
      {
        const double v_j = fe_v.shape_value(j, q);
        // Streaming term
        local_matrix(i, j) -= v_j * fe_v.shape_grad(i, q) * dir * JxW[q];
        // Loss term
        local_matrix(i, j) += u_i * v_j * material.sigma_t * JxW[q];
        // Accumulate scalar flux at this qp
        if (has_scattering)
          phi_old_q += local_phi_old[j] * v_j;
      }
      // Source gain term
      local_vector(i) += u_i * material.src * JxW[q];
      // Scattering gain term
      if (has_scattering)
        local_vector(i) += u_i * material.sigma_s * phi_old_q * JxW[q];
    }
  }
}

void
SNProblem::integrate_boundary(DoFInfo & dinfo, CellInfo & info, Point<2> dir)
{
}

void
SNProblem::integrate_face(
    DoFInfo & dinfo1, DoFInfo & dinfo2, CellInfo & info1, CellInfo & info2, Point<2> dir)
{
  const FEValuesBase<2> & fe1 = info1.fe_values();
  const FEValuesBase<2> & fe2 = info2.fe_values();
  const std::vector<double> & JxW = fe1.get_JxW_values();

  // Dot product between the direction and the outgoing normal of face 1
  const double dot = dir * fe1.normal_vector(0);

  // Cell 1 is outgoing
  if (dot > 0)
  {
    FullMatrix<double> & u1_v1_matrix = dinfo1.matrix(0, false).matrix;
    FullMatrix<double> & u1_v2_matrix = dinfo2.matrix(0, true).matrix;
    for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < fe1.dofs_per_cell; ++i)
      {
        const double u1_i = fe1.shape_value(i, q);
        for (unsigned int j = 0; j < fe1.dofs_per_cell; ++j)
          u1_v1_matrix(i, j) += dot * u1_i * fe1.shape_value(j, q) * JxW[q];
        for (unsigned int l = 0; l < fe2.dofs_per_cell; ++l)
          u1_v2_matrix(i, l) -= dot * u1_i * fe2.shape_value(l, q) * JxW[q];
      }
  }
  // Cell 2 is outgoing
  else if (dot < 0)
  {
    FullMatrix<double> & u2_v1_matrix = dinfo1.matrix(0, true).matrix;
    FullMatrix<double> & u2_v2_matrix = dinfo2.matrix(0, false).matrix;
    for (unsigned int q = 0; q < fe1.n_quadrature_points; ++q)
      for (unsigned int k = 0; k < fe2.dofs_per_cell; ++k)
      {
        const double u2_k = fe2.shape_value(k, q);
        for (unsigned int j = 0; j < fe1.dofs_per_cell; ++j)
          u2_v1_matrix(k, j) += dot * u2_k * fe1.shape_value(j, q) * JxW[q];
        for (unsigned int l = 0; l < fe2.dofs_per_cell; ++l)
          u2_v2_matrix(k, l) -= dot * u2_k * fe2.shape_value(l, q) * JxW[q];
      }
  }
}

void
SNProblem::solve_direction(unsigned int d)
{
  SolverControl solver_control(10000, 1e-10);
  SolverRichardson<> solver(solver_control);
  PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, fe.dofs_per_cell);
  solver.solve(system_matrix, solution, rhs, preconditioner);
}

void
SNProblem::solve()
{
  // Zero scalar flux
  phi = 0;

  for (unsigned int d = 0; d < aq.n_dir(); ++d)
  {
    std::cout << "Solving direction " << d << std::endl;

    // Assemble and solve
    assemble_direction(d);
    solve_direction(d);

    // Update scalar flux at each node
    phi.add(aq.w(d), solution);
  }

  phi_old = phi;
}

void
SNProblem::output_results() const
{
  std::cout << phi << std::endl;
  const std::string filename = "sol.gnuplot";
  deallog << "Writing solution to <" << filename << ">" << std::endl;
  std::ofstream gnuplot_output(filename.c_str());
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(phi, "phi");
  data_out.build_patches();
  data_out.write_gnuplot(gnuplot_output);
}

void
SNProblem::add_material(const unsigned int id, const Material & material)
{
  if (materials.find(id) != materials.end())
    throw ExcMessage("Material id already exists in material map");
  else
    materials.emplace(id, material);
}

void
SNProblem::run()
{
  setup_system();
  solve();
  output_results();
}
