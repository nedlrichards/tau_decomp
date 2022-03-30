"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

sec4 = Section()
stab_height = sec4.stable_cntr_height(sec4.lvls)
#stab_spice = sec4.stable_spice(sec4.lvls)
stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_lvls = sec4.stable_spice(stab_height)

prof_i = 252

z_a = sec4.z_a

plt_i = z_a <= 150.
c_field = sec4.c[plt_i, :]

prop_i = sec4.z_a <= 150.

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6,4))

ax[0].plot(sec4.sigma0[prop_i, prof_i], z_a[plt_i], 'k')
ax[0].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], '0.4')
ax[1].plot(sec4.spice[prop_i, prof_i], z_a[plt_i], 'k')
ax[1].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], '0.4')
ax[2].plot(c_field[:, prof_i], z_a[plt_i], 'k')
ax[2].plot([-10, 1e4], [sld_z[prof_i], sld_z[prof_i]], '0.4')

ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('$\sigma_0$ (kg/m$^3$)')
ax[1].set_xlabel(r'$\tau$ (kg/m$^3$)')
ax[2].set_xlabel('c (m/s)')

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[2].text(1499.5, sld_z[prof_i] - 7, 'Sonic layer depth', bbox=bbox)

ax[0].set_xlim(25.0, 26.0)
ax[1].set_xlim(1.1, 2.3)
ax[2].set_xlim(1499, 1513)
ax[2].set_ylim(150, 0)

pos = ax[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y1 += 0.06
pos.y0 += 0.02
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.02
pos.y1 += 0.06
pos.y0 += 0.02
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.02
pos.x1 += 0.04
pos.y1 += 0.06
pos.y0 += 0.02
ax[2].set_position(pos)

fig.savefig(join(savedir, 'sld_profile.png'), dpi=300)

