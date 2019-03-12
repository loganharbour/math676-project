#include "description.h"
#include "discretization.h"
#include "material.h"
#include "problem.h"

#include <deal.II/base/parameter_acceptor.h>
#include <iostream>

using namespace SNProblem;

int
main(int argc, char ** argv)
{
  try
  {
    // Initialize problem
    Problem problem;

    // Load the parameter file
    std::string parameter_file;
    // If a file is given via command-line argument, use it
    if (argc > 1)
      parameter_file = argv[1];
    // If not, use the default
    else
      parameter_file = "input.prm";
    ParameterAcceptor::initialize(parameter_file);

    // Run the problem and grab output
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
