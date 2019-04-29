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
fig, ax = plt.subplots(1)
fig.set_figwidth(8)
fig.set_figheight(3.5)
ax.plot(scat_inc[:,4], scat_inc[:,0], 'k-.', label='With scattering, incident BC')
ax.plot(scat_vac[:,4], scat_vac[:,0], 'k', label='With scattering, vacuum BC')
ax.plot(abs_inc[:,4], abs_inc[:,0], 'r--', label='Without scattering, incident BC')
ax.plot(abs_vac[:,4], abs_vac[:,0], 'r', label='Without scattering, vacuum BC')
ax.grid()
ax.set_xlabel('$x$')
ax.set_ylabel(r'$\phi(x, 5)$')
ax.legend()
fig.tight_layout()
fig.savefig('constant_solution.pdf')
