import numpy as np
import matplotlib.pyplot as plt

# Load sigma_s and times
result_dir = "results/"
ratios = [0.9, 0.99]
dsa_residuals = {}
without_residuals = {}
colors = {0.09 : 'g', 0.9 : 'r', 0.99 : 'k'}

# Load residuals
for ratio in ratios:
    dsa_residuals[ratio] = np.loadtxt(result_dir + "true_ratio_{}.csv".format(ratio))
    without_residuals[ratio] = np.loadtxt(result_dir + "false_ratio_{}.csv".format(ratio))

# Norm plot
fig, ax = plt.subplots(1)
for ratio in ratios:
    dsa = dsa_residuals[ratio]
    without = without_residuals[ratio]
    ax.semilogy(np.arange(len(dsa)) + 1, dsa, '--', linewidth=1.25,
                label='DSA, $c = {}$'.format(ratio), color=colors[ratio])
    ax.semilogy(np.arange(len(without)) + 1, without, linewidth=1.25,
                label='Without, $c = {}$'.format(ratio), color=colors[ratio])
ax.grid()
ax.legend()
ax.set_xlabel('Source iteration count')
ax.set_ylabel('Solution difference L$_2$ norm')
fig.tight_layout()
fig.savefig('residuals.pdf')
