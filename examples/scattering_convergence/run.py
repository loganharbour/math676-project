from shutil import copyfile
from timeit import timeit
import numpy as np

base_file = 'scattering_convergence.prm'
run_file = 'run.prm'

# ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for sigma_s in np.linspace(5, 95, num=19):
    times = open("results/times.csv", "a+")
    copyfile(base_file, run_file)
    lines = open(run_file, 'r').readlines()
    lines[2] = "  set material_sigma_s = {}\n".format(sigma_s)
    lines[15] = "  set residual_filename = results/sigma_s_{:d}\n".format(int(sigma_s))
    out = open(run_file, 'w')
    out.writelines(lines)
    out.close()
    time = timeit(stmt = "subprocess.run(['../../project', 'run.prm'])",
                  setup = "import subprocess", number = 1)
    times.write('{},{}\n'.format(sigma_s, time))
    times.close()
