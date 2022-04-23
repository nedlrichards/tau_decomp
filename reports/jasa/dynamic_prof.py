"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field, Config

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

cf = Config()

sec4 = Section()
stab_height = sec4.stable_cntr_height(sec4.lvls)
#stab_spice = sec4.stable_spice(sec4.lvls)
stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_lvls = sec4.stable_spice(stab_height)

#prof_i = np.arange(250, 255)
prof_i = np.array([249, 252])

section = 'tl_section_230.npz'
tl = np.load(join('data/processed/field_400', section))
proc_i = np.argmin(np.abs((tl['x_a'] / 1e3) - prof_i[:, None]), axis=-1)

z_a = sec4.z_a

plt_i = z_a <= 300.
c_field = sec4.c[plt_i, :]

prop_i = sec4.z_a <= 300.
proc_z_a = tl['z_a']

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(cf.jasa_2clm, 3))

ax[0].plot(sec4.sigma0[np.ix_(prop_i, prof_i)] - 25., z_a[plt_i])
ax[1].plot(sec4.spice[np.ix_(prop_i, prof_i)], z_a[plt_i])
ax[2].plot(c_field[:, prof_i] - 1500., z_a[plt_i])

#[a.set_prop_cycle(None) for a in ax]
#ax[0].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
#ax[1].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
ax[2].plot([-10, 1e4], [sld_z[prof_i], sld_z[prof_i]], color='0.6')

ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('$\sigma_0$ (kg/m$^3$)')
ax[1].set_xlabel(r'$\tau$ (kg/m$^3$)')
ax[2].text(7.5, 178.7, 'c (m/s)')
ax[2].text(11.7, 176.0, '+1500.')
ax[0].text(0.95, 176.0, '+25.')

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[2].legend([s + ' km' for s in map(str, list(prof_i))], loc=(0.03, 0.78))

ax[0].set_xlim(0, 1)
ax[1].set_xlim(1.1, 2.3)
ax[2].set_xlim(5, 12.5)
ax[2].set_ylim(150, 0)

pos = ax[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y1 += 0.07
pos.y0 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.02
pos.y1 += 0.07
pos.y0 += 0.06
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.02
pos.x1 += 0.04
pos.y1 += 0.07
pos.y0 += 0.06
ax[2].set_position(pos)

fig.savefig(join(savedir, 'sld_profile.png'), dpi=300)

