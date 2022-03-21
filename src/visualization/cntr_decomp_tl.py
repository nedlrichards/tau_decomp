"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from copy import copy
import os

from src import sonic_layer_depth

#plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')

fc = 400

def plot_comp(section):
    tl_data = np.load(os.path.join(f'data/processed/field_{int(fc)}', section))
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

    fig, axes = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(7.5, 8.00))
    vmax = np.ma.max(c_sld)
    vmin = np.ma.min(c_sld)
    x_t = x_a[0] / 1e3 - 1
    bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
    cm = axes[0, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_bg[z_i, :],
                               cmap=plt.cm.coolwarm,
                               vmax=vmax, vmin=vmin,
                               rasterized=True)
    axes[0,1].text(x_t, 0, 'Background', bbox=bbox, zorder=50, ha='center')
    cm = axes[1, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_tilt[z_i, :],
                               cmap=plt.cm.coolwarm,
                               vmax=vmax, vmin=vmin,
                               rasterized=True)
    axes[1,1].text(x_t, 0, 'Tilt', bbox=bbox, zorder=50, ha='center')
    cm = axes[2, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_spice[z_i, :],
                               cmap=plt.cm.coolwarm,
                               vmax=vmax, vmin=vmin,
                               rasterized=True)
    axes[2,1].text(x_t, 0, 'Spice', bbox=bbox, zorder=50, ha='center')
    cm = axes[3, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_total[z_i, :],
                               cmap=plt.cm.coolwarm,
                               vmax=vmax, vmin=vmin,
                               rasterized=True)
    axes[3,1].text(x_t, 0, 'Measured', bbox=bbox, zorder=50, ha='center')

    cax = fig.add_axes([0.12, 0.93, 0.40, 0.03])
    cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
    cbar.set_label('Sound speed (m/s)')
    loc = plticker.MaxNLocator(nbins=4, integer=True)
    cbar.ax.xaxis.set_major_locator(loc)

    cm = axes[0, 1].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(p_bg)).T,
                            cmap=cmap, vmax=-50, vmin=-90, rasterized=True)
    cm = axes[1, 1].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(p_tilt)).T,
                            cmap=cmap, vmax=-50, vmin=-90, rasterized=True)
    cm = axes[2, 1].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(p_spice)).T,
                            cmap=cmap, vmax=-50, vmin=-90, rasterized=True)
    cm = axes[3, 1].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(p_total)).T,
                            cmap=cmap, vmax=-50, vmin=-90, rasterized=True)

    cax = fig.add_axes([0.55, 0.93, 0.40, 0.03])
    cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
    cbar.set_label('Acoustic pressure (dB re 1m)')
    cbar.set_ticks(cbar.get_ticks()[1:])

    #fig.supxlabel('Range (km)')
    #fig.supylabel('Depth (m)')
    axes[0, 0].set_ylim(150, 0)
    axes[0, 0].set_xlim(rplot[0] / 1e3, rplot[-1] / 1e3)

    pos = axes[0, 0].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[0, 0].set_position(pos)

    pos = axes[1, 0].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[1, 0].set_position(pos)

    pos = axes[2, 0].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[2, 0].set_position(pos)

    pos = axes[3, 0].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[3, 0].set_position(pos)


    pos = axes[0, 1].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[0, 1].set_position(pos)

    pos = axes[1, 1].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[1, 1].set_position(pos)

    pos = axes[2, 1].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[2, 1].set_position(pos)

    pos = axes[3, 1].get_position()
    pos.x0 -= 0.01
    pos.x1 += 0.05
    pos.y0 -= 0.02
    pos.y1 -= 0.02
    axes[3, 1].set_position(pos)

    x_s = int(tl_data['xs']/1e3)

    fig.savefig(f'reports/figures/decomp_{int(fc)}/' + f'decomp_section_{x_s}km.png', dpi=300)
    plt.close(fig)

#for sec in filter(lambda x: len(x.split('.')) == 2, os.listdir(f'data/processed/field_{int(fc)}')):
sec = 'xmission_000.npz'
plot_comp(sec)

