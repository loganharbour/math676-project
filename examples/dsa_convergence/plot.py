import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 12.0

# Load sigma_s and times
result_dir = "results/"
ratios = [0.09, 0.9, 0.99]
dsa_residuals = {}
without_residuals = {}
colors = {0.09 : 'k', 0.9 : 'r', 0.99 : 'g'}

# Load residuals
for ratio in ratios:
    dsa_residuals[ratio] = np.loadtxt(result_dir + "true_ratio_{}.csv".format(ratio))
    without_residuals[ratio] = np.loadtxt(result_dir + "false_ratio_{}.csv".format(ratio))

# Norm plot
fig, ax = plt.subplots(1, 2)
fig.set_figwidth(8)
fig.set_figheight(3.5)
for ratio in ratios:
    dsa = dsa_residuals[ratio]
    without = without_residuals[ratio]
    ax[0].semilogy(np.arange(len(dsa)) + 1, dsa, '--', linewidth=1.25,
                label='DSA, $c = {}$'.format(ratio), color=colors[ratio])
    ax[0].semilogy(np.arange(len(without)) + 1, without, linewidth=1.25,
                label='Without DSA, $c = {}$'.format(ratio), color=colors[ratio])
    ax[1].loglog(np.arange(len(dsa)) + 1, dsa, '--', linewidth=1.25,
                label='DSA, $c = {}$'.format(ratio), color=colors[ratio])
    ax[1].loglog(np.arange(len(without)) + 1, without, linewidth=1.25,
                label='Without DSA, $c = {}$'.format(ratio), color=colors[ratio])
ax[0].grid()
ax[1].grid()
ax[0].set_ylabel('Solution difference L$_2$ norm')
ax[0].set_xlabel('Source iteration count')
ax[1].set_xlabel('Source iteration count')
handles, labels = ax[0].get_legend_handles_labels()
lgd = ax[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(1.0, -0.4),
                   ncol=3, fontsize=9)
fig.tight_layout()
fig.savefig('dsa_residuals.pdf', bbox_inches='tight', bbox_extra_artists=(lgd,))
