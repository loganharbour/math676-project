from shutil import copyfile
from timeit import timeit
import numpy as np

base_file = '../constant_with_scattering.prm'
run_file = 'run.prm'
runs_per = 3

results = []

for refinement in [5, 6, 7, 8, 9]:
    print('Refinement {}'.format(refinement))
    result = []
    for renumber in ['true', 'false']:
        print('  Renumber = {}'.format(renumber))
        average = 0
        for i in range(runs_per):
            copyfile(base_file, run_file)
            lines = open(run_file, 'r').readlines()
            lines[16] = "  set renumber = {}\n".format(renumber)
            lines[17] = "  set uniform_refinement = {}\n".format(refinement)
            out = open(run_file, 'w')
            out.writelines(lines)
            out.close()
            time = timeit(stmt = "subprocess.run(['../../project', 'run.prm'], stdout=subprocess.DEVNULL)",
                          setup = "import subprocess", number = 1)
            average += time / runs_per
            print('    Run {} time = {:.2f} sec'.format(i, time))
        result.append(average)
    results.append([refinement, result[0], result[1]])

np.savetxt('results.csv', results, delimiter=',', header='refinements, with, without')
