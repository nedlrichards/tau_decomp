"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from copy import copy
import os

from src import sonic_layer_depth, list_tl_files, section_cfield, Config

#plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')

fc = 400
#fc = 1e3
source_depth = 'shallow'
cf = Config(fc=fc, source_depth=source_depth)

save_dir = f'reports/jasa/figures/decomp_{int(fc)}_' + source_depth

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']


def plot_comp(tl_file):
    tl_data = np.load(tl_file)
    zplot = tl_data['zplot']
    z_a = tl_data['z_a']
    rplot = tl_data['rplot']
    p_bg = tl_data['p_bg']
    p_tilt = tl_data['p_tilt']
    p_spice = tl_data['p_spice']
    p_total = tl_data['p_total']

    x_sec, c_bg_sec = section_cfield(tl_data['xs'], x_a, c_bg, rmax=cf.rmax)
    _, c_tilt_sec = section_cfield(tl_data['xs'], x_a, c_tilt, rmax=cf.rmax)
    _, c_spice_sec = section_cfield(tl_data['xs'], x_a, c_spice, rmax=cf.rmax)
    _, c_total_sec = section_cfield(tl_data['xs'], x_a, c_total, rmax=cf.rmax)

    c_tilt_sec -= c_bg_sec
    c_spice_sec -= c_bg_sec
    c_total_sec -= c_bg_sec

    z_i = z_a < 350.

    sld_z, _ = sonic_layer_depth(z_a[z_i], c_bg_sec[z_i, :])
    sld_m = z_a[z_i, None] > sld_z
    c_sld = np.ma.array(c_bg_sec[z_i, :], mask=sld_m)

    c_plot_ref = np.mean(c_sld)

    fig, axes = plt.subplots(4, 2, sharey=True, sharex=True, figsize=(7.5, 8.00))
    vmax = np.ma.max(c_sld)
    vmin = np.ma.min(c_sld)
    x_t = x_sec[0] / 1e3 - 1
    bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
    cm0 = axes[0, 0].pcolormesh(x_sec / 1e3, z_a[z_i], c_bg_sec[z_i, :],
                               cmap=plt.cm.coolwarm,
                               vmax=vmax, vmin=vmin,
                               rasterized=True)
    axes[0,1].text(x_t, 0, 'Background', bbox=bbox, zorder=50, ha='center')
    cm1 = axes[1, 0].pcolormesh(x_sec / 1e3, z_a[z_i], c_tilt_sec[z_i, :],
                               cmap=plt.cm.BrBG,
                               vmin=-3, vmax=3,
                               rasterized=True)
    axes[1,1].text(x_t, 0, 'Tilt', bbox=bbox, zorder=50, ha='center')
    cm = axes[2, 0].pcolormesh(x_sec / 1e3, z_a[z_i], c_spice_sec[z_i, :],
                               cmap=plt.cm.BrBG,
                               vmin=-3, vmax=3,
                               rasterized=True)
    axes[2,1].text(x_t, 0, 'Spice', bbox=bbox, zorder=50, ha='center')
    cm = axes[3, 0].pcolormesh(x_sec / 1e3, z_a[z_i], c_total_sec[z_i, :],
                               cmap=plt.cm.BrBG,
                               vmin=-3, vmax=3,
                               rasterized=True)
    axes[3,1].text(x_t, 0, 'Measured', bbox=bbox, zorder=50, ha='center')

    cax = fig.add_axes([0.12, 0.93, 0.40, 0.03])
    cbar = fig.colorbar(cm0, cax=cax, orientation='horizontal')

    cax = fig.add_axes([0.12, 0.93, 0.40, 0.03])
    cbar = fig.colorbar(cm1, cax=cax, orientation='horizontal')

    cbar.set_label('Sound speed (m/s)')
    loc = plticker.MaxNLocator(nbins=4, integer=True)
    cbar.ax.xaxis.set_major_locator(loc)
    cbar.ax.xaxis.get_major_formatter().set_useOffset(False)

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
    axes[0, 0].set_ylim(350, 0)
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

    fig.savefig(os.path.join(save_dir, f'decomp_section_{x_s}km.png'), dpi=300)
    plt.close(fig)

for tl in list_tl_files(fc, source_depth=source_depth):
    plot_comp(tl)
    break


