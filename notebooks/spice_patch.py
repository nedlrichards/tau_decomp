"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from src import Section
from src.data import lp_incompute

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

sec4 = Section()
starting_spice = sec4.lvls.copy()
filled_spice = sec4.lvls.copy()

min_x = 400
max_x = 980
#call_lvls = [35, 39, 43]
call_lvls = np.array([35, 36, 37]) - 18

last_full_i = np.argmax(~np.any(np.isnan(filled_spice[1, :, :]), axis=1))
for i in range(last_full_i + 1, call_lvls[0] - 1, -1):
    patched_lvl = lp_incompute.patch_spice(i, filled_spice)
    filled_spice[1, i, :] = patched_lvl

fig, ax = plt.subplots()
ax.plot(sec4.x_a / 1e3, filled_spice[1, call_lvls, :].T, 'k')
ax.plot(sec4.x_a / 1e3, starting_spice[1, call_lvls, :].T)

ax.set_ylim(1.8, 2.5)
ax.set_xlim(min_x, max_x)


1/0

_, tau = grid_field(sec4.z_a, sec4.lvls, sec4.sig_lvl)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5))
x_i = sec4.x_a < max_x * 1e3
z_i = sec4.z_a <= 150.

cc0 = ['C0', 'C1', '0.2']
for i, (c0, lbl_i) in enumerate(zip(cc0, call_lvls)):
    plt_height = stab_height[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    ax[0].plot(sec4.x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119')
    ax[0].plot(sec4.x_a / 1e3, sec4.lvls[0, lbl_i, :].T, color=c0)

    #z_off = 10 if lbl_i == 54 else 0
    z_off = (i - 1) * 5
    if i == 1: z_off -= 5

    ax[0].text(max_x + 3., sec4.lvls[0, lbl_i, x_i][-1] + z_off,
               f'{sec4.sig_lvl[lbl_i]:.2f}', va='center', color=c0)

pos = ax[0].get_position()
pos.x1 -= 0.08
pos.x0 += 0.05
pos.y1 += 0.08
pos.y0 += 0.06
ax[0].set_position(pos)

ax[0].set_ylabel('Depth (m)')
ax[0].text(max_x + 3., 30, '$\sigma$', va='center')
ax[0].text(max_x + 3., 50, '(kg/m$^3$)', va='center')

pos = ax[1].get_position()
pos.x1 -= 0.08
pos.x0 += 0.05
pos.y1 += 0.06
pos.y0 += 0.04
ax[1].set_position(pos)

ax[1].set_ylabel(r'$\tau$ (kg/m$^3$)')

for c0, lbl_i in zip(cc0, call_lvls):
    ax[1].plot(sec4.x_a / 1e3, sec4.lvls[1, lbl_i, :].T, color=c0)
    ax[1].text(max_x + 3., sec4.lvls[1, lbl_i, x_i][-1],
               f'{sec4.sig_lvl[lbl_i]:.2f}', color=c0, va='center')

ax[1].set_xlabel('Range (km)')

ax[1].set_ylim(1.8, 2.3)

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
#ax[0].xaxis.set_ticks_position('bottom')
#ax[0].yaxis.set_ticks_position('left')

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
#ax[1].xaxis.set_ticks_position('bottom')
#ax[1].yaxis.set_ticks_position('left')

#fig.savefig(join(savedir, 'sig_tau_interp.png'), dpi=300)

