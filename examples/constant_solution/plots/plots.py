import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'cm'
plt.rcParams['font.size'] = 12.0

scat_inc = np.loadtxt('scattering_with_incident.csv', delimiter=',', skiprows=1)
scat_vac = np.loadtxt('scattering_with_vacuum.csv', delimiter=',', skiprows=1)
abs_inc = np.loadtxt('absorber_with_incident.csv', delimiter=',', skiprows=1)
abs_vac = np.loadtxt('absorber_with_vacuum.csv', delimiter=',', skiprows=1)

# Norm plot
fig, ax = plt.subplots(1, 2)
fig.set_figwidth(8)
fig.set_figheight(3.5)
ax[0].plot(scat_vac[:,4], scat_vac[:,0] / 10, 'r', label='Vacuum BCs')
ax[0].plot(scat_inc[:,4], scat_inc[:,0] / 10, 'k--', label='Incident BCs')
ax[0].set_title(r'$\sigma_s = 9.9$', fontsize=12)
ax[1].plot(abs_vac[:,4], abs_vac[:,0] / 0.1, 'r', label='Vacuum BCs')
ax[1].plot(abs_inc[:,4], abs_inc[:,0] / 0.1, 'k--', label='Incident BCs')
ax[1].set_title(r'$\sigma_s = 0.0$', fontsize=12)
ax[0].grid()
ax[1].grid()
ax[0].set_xlabel('$x$')
ax[1].set_xlabel('$x$')
ax[0].set_ylabel(r'Normalized $\phi(x, 8)$')
ax[0].legend()
ax[1].legend()
fig.tight_layout()
fig.savefig('constant_solution.pdf')
