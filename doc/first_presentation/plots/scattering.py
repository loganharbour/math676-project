import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 12.0

# Load sigma_s and times
result_dir = "../../../examples/scattering_convergence/results/"
data = np.loadtxt(result_dir + "times.csv", delimiter=',')
sigma_s = data[:, 0]
times = data[:, 1]
ratios = sigma_s / 100
# Load residuals
residuals = []
for sig in sigma_s:
    residuals.append(np.loadtxt(result_dir + "sigma_s_{:d}.csv".format(int(sig))))

# Norm plot
fig, ax = plt.subplots(1)
fig.set_figwidth(4)
fig.set_figheight(4)
for i in list(range(len(sigma_s)))[2::4]:
    ax.semilogy(np.arange(len(residuals[i])) + 1, residuals[i], linewidth=1.25,
                label=r'$\sigma_s / \sigma_t = {:.2f}$'.format(ratios[i]))
ax.grid()
ax.legend()
ax.set_xlabel('Source iteration count')
ax.set_ylabel('Solution difference L$_2$ norm')
fig.tight_layout()
fig.savefig('scattering_norms.pdf')

# Norm plot
fig, ax = plt.subplots(1)
fig.set_figwidth(4)
fig.set_figheight(4)
ax.plot(ratios, times / np.min(times), 'k.--', linewidth=0.75)
ax.grid()
ax.set_xlabel(r'Scattering ratio, $\sigma_s / \sigma_t$')
ax.set_ylabel('Normalized computation time')
fig.tight_layout()
fig.savefig('scattering_times.pdf')
