#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <iostream>

using namespace SNProblem;

int
main()
{
  try
  {
    Description description;
    description.add_material(0, Material(0.1, 0.0, 1.0));

    Discretization discretization;

    Problem problem(description, discretization);
    problem.run();
  }
  catch (std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
