"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from copy import copy
import os

from src import sonic_layer_depth, list_tl_files

plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')

fc = 400
tl_files = list_tl_files(fc)
#tl_file = tl_files[1]
tl_file = tl_files[23]

tl_data = np.load(tl_file)
zplot = tl_data['zplot']
z_a = tl_data['z_a']
x_a = tl_data['x_a']
rplot = tl_data['rplot']
p_bg = tl_data['p_bg']
p_tilt = tl_data['p_tilt']
p_spice = tl_data['p_spice']
p_total = tl_data['p_total']
c_bg = tl_data['c_bg']
c_tilt = tl_data['c_tilt']
c_spice = tl_data['c_spice']
c_total = tl_data['c_total']


z_i = z_a < 150.

sld_z, _ = sonic_layer_depth(z_a[z_i], c_total[z_i, :])
sld_m = z_a[z_i, None] > sld_z
c_sld = np.ma.array(c_total[z_i, :], mask=sld_m)

c_plot_ref = np.mean(c_sld)

vmax = np.ma.max(c_sld)
vmin = np.ma.min(c_sld)
x_t = x_a[0] / 1e3 - 1

bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7.5, 3.00))
cm = axes[0].pcolormesh(x_a / 1e3, z_a[z_i], c_total[z_i, :],
                            cmap=plt.cm.coolwarm,
                            vmax=vmax, vmin=vmin,
                            rasterized=True)

cax = fig.add_axes([0.12, 0.94, 0.40, 0.03])
cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
cbar.set_label('Sound speed (m/s)')
loc = plticker.MaxNLocator(nbins=4, integer=True)
cbar.ax.xaxis.set_major_locator(loc)

cm = axes[1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_total)).T,
                        cmap=cmap, vmax=-50, vmin=-90, rasterized=True)

cax = fig.add_axes([0.55, 0.94, 0.40, 0.03])
cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
cbar.set_label('Acoustic pressure (dB re 1m)')
cbar.set_ticks(cbar.get_ticks()[1:])
fig.supxlabel('Postion, $x$ (km)')
axes[0].set_ylabel('Depth (m)')

axes[0].set_ylim(150, 0)
axes[0].set_xlim(rplot[0] / 1e3, rplot[-1] / 1e3)

pos = axes[0].get_position()
pos.x0 -= 0.01
pos.x1 += 0.05
pos.y0 += 0.04
pos.y1 -= 0.09
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 -= 0.01
pos.x1 += 0.05
pos.y0 += 0.04
pos.y1 -= 0.09
axes[1].set_position(pos)


x_s = int(tl_data['xs']/1e3)

fig.savefig(f'reports/spice_po/figures/decomp_section_{x_s}km.png', dpi=300)

