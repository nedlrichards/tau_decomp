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

min_x = 230
max_x = 350
call_lvls = [34, 38, 42, 48, 52]

#min_x = 200
#max_x = 400
#call_lvls = [38, 41, 50]

fig, ax = plt.subplots(sharex=True, figsize=(cf.jasa_1clm, 2))
x_i = x_a < max_x * 1e3

cc0 = ['0.6', '0.5', '0.4', '0.3', '0.2']
z_off = [60, 0, 0, 10, 20]
for zo, c0, lbl_i in zip(z_off, cc0, call_lvls):
    plt_height = stable_lvls[0, lbl_i, :]
    plt_inds = plt_height > 1e-5
    ax.plot(x_a[plt_inds] / 1e3, plt_height[plt_inds], color='#be0119', alpha=0.6, linewidth=1)
    plt_lvls = lvls[0, lbl_i, :]
    plt_lvls[np.isnan(plt_lvls)] = -2
    ax.plot(x_a / 1e3, plt_lvls, color=c0)

    ax.text(max_x + 5., lvls[0, lbl_i, x_i][-1] + zo,
               f'{sigma[lbl_i]:.2f}', va='center', color=c0)

ax.set_ylim(130, 0)
ax.set_xlim(min_x, max_x)
ax.set_xticks([230, 250, 270, 290, 310, 330])
ax.set_xticks([240, 260, 280, 300, 320, 340], minor=True)
ax.xaxis.set_ticks_position('both')

pos = ax.get_position()
pos.x1 -= 0.07
pos.x0 += 0.08
pos.y1 += 0.08
pos.y0 += 0.13
ax.set_position(pos)

ax.set_ylabel('Depth (m)')
ax.set_xlabel('Transcet range, $x$ (km)')
ax.text(max_x + 1., 10, '$\sigma$', va='center')
ax.text(max_x + 1., 30, '(kg/m$^3$)', va='center')

fig.savefig(join(savedir, 'sig_tau_interp.png'), dpi=300)
