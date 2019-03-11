#include "problem.h"

#include <deal.II/numerics/data_out.h>
#include <fstream>

namespace SNProblem
{
using namespace dealii;

double
Problem::L2_difference(const Vector<double> & v1, const Vector<double> & v2)
{
  double value = 0;

  const auto cell_worker = [&](DoFInfo & dinfo, CellInfo & info) {
    const FEValuesBase<2> & fe_v = info.fe_values();
    const std::vector<double> & JxW = fe_v.get_JxW_values();

    for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
    {
      const double diff_i = v1[dinfo.indices[i]] - v2[dinfo.indices[i]];
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        value += std::pow(fe_v.shape_value(i, q) * diff_i, 2) * JxW[q];
    }
  };

  // Call loop to execute the integration
  MeshWorker::DoFInfo<2> dof_info(dof_handler);
  MeshWorker::loop<2, 2, MeshWorker::DoFInfo<2>, MeshWorker::IntegrationInfoBox<2>>(
      dof_handler.begin_active(),
      dof_handler.end(),
      dof_info,
      discretization.info_box,
      cell_worker,
      NULL,
      NULL,
      assembler);

  return value;
}

void
Problem::output() const
{
  std::ofstream output("solution.vtu");
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(scalar_flux, "scalar_flux");
  data_out.build_patches();
  data_out.write_vtu(output);
}

}
