import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 12.0

# Load sigma_s and times
data = np.loadtxt("../../../examples/renumbering/results.csv", delimiter=',')
refinements = data[:, 0] - 5
ratios = data[:, 2] / data[:, 1]

# Norm plot
fig, ax = plt.subplots(1)
fig.set_figwidth(4)
fig.set_figheight(4)
ax.plot(refinements, ratios, 'k.--', linewidth=0.75)
ax.grid()
ax.set_xlabel('Uniform refinements')
ax.set_ylabel('Normalized time (without / with)')
fig.tight_layout()
fig.savefig('renumbering.pdf')
