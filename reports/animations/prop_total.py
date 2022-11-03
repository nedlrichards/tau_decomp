"""Compare results of gridded and reconstructed total field"""

# ffmpeg -r 5 -i reports/animations/stills/xmitt_%03d.png -c:v libx264 reports/animations/output.mp4

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.transforms
from copy import copy
import os

from src import sonic_layer_depth, list_tl_files, Config, section_cfield

#plt.ion()
cf = Config()

fc = 400
savedir = 'reports/animations/stills'
load_dir = 'reports/animations/processed/field_400_shallow'

fields = np.load('data/processed/inputed_decomp.npz')
tl_files = os.listdir(load_dir)
tl_files.sort(key=lambda tl: int(tl.split('_')[-1].split('.')[0]))

x_c_a, c_total = section_cfield(0, fields['x_a'], fields['c_total'], rmax=360e3)
for tf in tl_files:
    tl_data = np.load(os.path.join(load_dir, tf))
    zplot = tl_data['zplot']
    z_a = tl_data['z_a']
    x_a = tl_data['x_a']
    xs = tl_data['xs']
    rplot = tl_data['rplot']
    p_total = tl_data['p_bg']

    z_i = zplot < 200.

    fig, axes = plt.subplots(2, 1, sharey=True, figsize=(9, 5))
    ax = axes[0]
    x_t = x_a[0] / 1e3 - 2
    cm0 = ax.pcolormesh(rplot / 1e3, zplot[z_i],
                        20 * np.log10(np.abs(p_total[:, z_i].T)),
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

    cax = fig.add_axes([.90, .58, 0.02, 0.35])

    cbar = fig.colorbar(cm0, cax=cax)
    cbar.set_label('pressure (dB re 1 m)')
    cbar.set_ticks([-50, -70, -90])

    ax.set_yticks([0, 200])
    ax.set_xticks([xs / 1e3, xs / 1e3 + 30, xs / 1e3 + 60])

    fig.supylabel('Depth (m)')
    ax.set_ylim(200, 0)

    pos = ax.get_position()
    pos.x0 += -0.04
    pos.x1 += -0.02
    pos.y0 += 0.04
    pos.y1 += 0.06
    ax.set_position(pos)

    ax = axes[1]
    cb = ax.pcolormesh(x_c_a / 1e3, z_a, c_total, cmap=plt.cm.coolwarm, vmin=1504, vmax=1512)
    cax = fig.add_axes([.90, .10, 0.02, 0.35])

    cbar = fig.colorbar(cb, cax=cax)
    cbar.set_label('Sound speed (m/s)')
    ax.set_xlabel('Range (km)')
    cbar.set_ticks([1504, 1508, 1512])

    pos = ax.get_position()
    pos.x0 += -0.04
    pos.x1 += -0.02
    pos.y0 += -0.02
    pos.y1 += 0.00
    ax.set_position(pos)

    ax.plot([xs / 1e3, xs / 1e3 + 60], [175, 175], color='C4')
    ax.plot([xs / 1e3, xs / 1e3 + 60], [175, 175], 'x', color='C4')


    fig.savefig(os.path.join(savedir, f'xmitt_{int(xs/1e3):03}.png'), dpi=300)
    plt.close(fig)
