"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/spice_po/figures'

sec4 = Section()
stab_height = sec4.stable_cntr_height(sec4.lvls)
#stab_spice = sec4.stable_spice(sec4.lvls)
stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_lvls = sec4.stable_spice(stab_height)

z_a = sec4.z_a

plt_i = z_a <= 150.
c_field = sec4.c[plt_i, :]

prop_i = sec4.z_a <= 150.

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)

# plot sigma, tau
#min_x = 50
#max_x = 250
#call_lvls = [39, 48, 54]

min_x = 200
max_x = 300
call_lvls = [37, 39, 41]

_, tau = grid_field(sec4.z_a, sec4.lvls, sec4.sig_lvl)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3))
x_i = sec4.x_a < max_x * 1e3
z_i = sec4.z_a <= 150.

cc0 = ['C0', 'C1', '0.2']
z_off = [0, 0, 0]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    plt_height = stab_height[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    #ax[0].plot(sec4.x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119')
    ax[0].plot(sec4.x_a / 1e3, sec4.lvls[0, lbl_i, :].T, color=c0)

    ax[0].text(max_x + 3., sec4.lvls[0, lbl_i, x_i][-1] + zo,
               f'{sec4.sig_lvl[lbl_i]:.2f}', va='center', color=c0)

ax[0].set_ylim(130, 0)
ax[0].set_xlim(min_x, max_x)

pos = ax[0].get_position()
pos.x1 -= 0.07
pos.x0 += 0.08
pos.y1 += 0.08
pos.y0 += 0.06
ax[0].set_position(pos)

ax[0].set_ylabel('Depth (m)')
ax[0].text(max_x + 3., 10, '$\sigma$', va='center')
ax[0].text(max_x + 3., 30, '(kg/m$^3$)', va='center')

pos = ax[1].get_position()
pos.x1 -= 0.07
pos.x0 += 0.08
pos.y1 += 0.06
pos.y0 += 0.04
ax[1].set_position(pos)

ax[1].set_ylabel(r'$\tau$ (kg/m$^3$)')

z_off = [0.0, 0.1, 0]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    ax[1].plot(sec4.x_a / 1e3, sec4.lvls[1, lbl_i, :].T, color=c0)
    ax[1].text(max_x + 3., sec4.lvls[1, lbl_i, x_i][-1] + zo,
               f'{sec4.sig_lvl[lbl_i]:.2f}', color=c0, va='center')

ax[1].set_xlabel('Range (km)')

ax[1].set_ylim(1.45, 2.25)

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
#ax[0].xaxis.set_ticks_position('bottom')
#ax[0].yaxis.set_ticks_position('left')

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
#ax[1].xaxis.set_ticks_position('bottom')
#ax[1].yaxis.set_ticks_position('left')

fig.savefig(join(savedir, 'sig_tau_300.png'), dpi=300)
