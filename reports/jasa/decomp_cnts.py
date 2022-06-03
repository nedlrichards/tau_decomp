"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Config

plt.ion()
cf = Config()
savedir = 'reports/jasa/figures'

inputed = np.load('data/processed/inputed_decomp.npz')

lvls  = inputed['filled_lvls']
stable_lvls = inputed['stable_spice_lvls']

sigma = inputed['sig_lvls']
x_a = inputed['x_a']

# plot sigma, tau
#min_x = 50
#max_x = 250
#call_lvls = [39, 48, 54]

min_x = 175
max_x = 400
call_lvls = [38, 42, 48]

#min_x = 200
#max_x = 400
#call_lvls = [38, 41, 50]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(cf.jasa_1clm, 3))
x_i = x_a < max_x * 1e3

cc0 = ['0.6', '0.4', '0.2']
z_off = [-10, -5, 0]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    plt_height = stable_lvls[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    ax[0].plot(x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119', alpha=0.6)
    ax[0].plot(x_a / 1e3, lvls[0, lbl_i, :].T, color=c0)

    ax[0].text(max_x + 3., lvls[0, lbl_i, x_i][-1] + zo,
               f'{sigma[lbl_i]:.2f}', va='center', color=c0)

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

ax[1].set_ylabel(r'$\gamma$ (kg/m$^3$)', labelpad=-0.05)

z_off = [-0.03, -0.05, 0.04]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    ax[1].plot(x_a / 1e3, lvls[1, lbl_i, :].T, color=c0)
    ax[1].text(max_x + 3., lvls[1, lbl_i, x_i][-1] + zo,
               f'{sigma[lbl_i]:.2f}', color=c0, va='center')

ax[1].plot(x_a / 1e3, stable_lvls[1, call_lvls[-1], :],color='#be0119', alpha=0.6)

ax[1].set_xlabel('Range (km)')

ax[1].set_ylim(-0.25, 0.3)

ax[0].text(min_x - 25., 5, '(a)', bbox=cf.bbox)
ax[1].text(min_x - 25., 0.30, '(b)', bbox=cf.bbox)


ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
#ax[0].xaxis.set_ticks_position('bottom')
#ax[0].yaxis.set_ticks_position('left')

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
#ax[1].xaxis.set_ticks_position('bottom')
#ax[1].yaxis.set_ticks_position('left')

fig.savefig(join(savedir, 'sig_tau_interp.png'), dpi=300)

