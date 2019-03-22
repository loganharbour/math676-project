from shutil import copyfile
import subprocess

base_file = 'scattering_convergence.prm'
run_file = 'run.prm'

for ratio in [0.9, 0.99]:
    for enabled in ['true', 'false']:
        copyfile(base_file, run_file)
        lines = open(run_file, 'r').readlines()
        lines[2] = "  set material_sigma_s = {}\n".format(100 * ratio)
        lines[16] = "  set residual_filename = results/{}_ratio_{}.csv\n".format(enabled, ratio)
        lines[22] = "  set enabled = {}\n".format(enabled)
        out = open(run_file, 'w')
        out.writelines(lines)
        out.close()
        subprocess.run(['../../project', 'run.prm'])
