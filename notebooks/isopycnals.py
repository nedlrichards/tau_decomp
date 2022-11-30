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

min_x = 200
max_x = 300
call_lvls = [34, 36, 42, 44]

#min_x = 200
#max_x = 400
#call_lvls = [38, 41, 50]

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2))
x_i = x_a < max_x * 1e3


for lbl_i in call_lvls:
    ax.plot(x_a / 1e3, lvls[0, lbl_i, :].T, color='k', linewidth=1)
    plt_height = stable_lvls[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    ax.plot(x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119', alpha=0.6)

"""
cc0 = ['0.6', '0.4', '0.2']
z_off = [-10, -5, 0]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    plt_height = stable_lvls[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    ax.plot(x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119', alpha=0.6)
    ax.plot(x_a / 1e3, lvls[0, lbl_i, :].T, color=c0)

    ax.text(max_x + 3., lvls[0, lbl_i, x_i][-1] + zo,
               f'{sigma[lbl_i]:.2f}', va='center', color=c0)

"""
ax.set_ylim(130, 0)
ax.set_xlim(min_x, max_x)

pos = ax.get_position()
pos.x1 -= 0.07
pos.x0 += 0.10
pos.y1 += 0.08
pos.y0 += 0.06
ax.set_position(pos)

ax.set_ylabel('Depth (m)')
ax.text(max_x - 3., 10, '$\sigma$', va='center')
ax.text(max_x - 3., 30, '(kg/m$^3$)', va='center')

ax.text(min_x - 25., 5, '(a)', bbox=cf.bbox)


ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
#ax[0].xaxis.set_ticks_position('bottom')
#ax[0].yaxis.set_ticks_position('left')

fig.savefig(join(savedir, 'sig_tau_interp.png'), dpi=300)
