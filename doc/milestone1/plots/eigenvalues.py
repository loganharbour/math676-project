import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 12.0

SI_func = lambda k: np.arctan(k) / k
DSA_func = lambda k: np.arctan(k) / k - 3 * (1 - np.arctan(k) / k) / k**2
ks = np.linspace(1e-6, 50, num=200)

fig, ax = plt.subplots(1)
fig.set_figwidth(4)
fig.set_figheight(2.6)
ax.plot(ks, SI_func(ks), 'k', label='Source iteration')
ax.plot(ks, DSA_func(ks), 'r', label='Diffusion acceleration')
ax.grid()
ax.set_xlabel(r'$\kappa = \lambda / \sigma_t$')
ax.set_ylabel('Largest eigenvalue')
ax.legend()
fig.tight_layout()
fig.savefig('eigenvalues.pdf')
