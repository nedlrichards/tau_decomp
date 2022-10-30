"""Compare results of gridded and reconstructed total field"""

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

for tf in tl_files:
    tl_data = np.load(os.path.join(load_dir, tf))
    zplot = tl_data['zplot']
    z_a = tl_data['z_a']
    x_a = tl_data['x_a']
    xs = tl_data['xs']
    rplot = tl_data['rplot']
    p_total = tl_data['p_bg']

    _, c_total = section_cfield(tl_data['xs'], fields['x_a'], fields['c_total'])


    z_i = zplot < 200.

    fig, ax = plt.subplots()
    x_t = x_a[0] / 1e3 - 2
    cm0 = ax.pcolormesh(rplot / 1e3, zplot[z_i],
                        20 * np.log10(np.abs(p_total[:, z_i].T)),
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

    cbar = fig.colorbar(cm0)
    cbar.set_label('pressure (dB re 1 m)')
    cbar.set_ticks([-50, -70, -90])

    ax.set_yticks([0, 200])
    ax.set_xticks([xs / 1e3, xs / 1e3 + 30, xs / 1e3 + 60])

    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(200, 0)

    pos = ax.get_position()
    pos.x0 += 0.02
    pos.x1 += 0.02
    pos.y0 += 0.04
    pos.y1 += 0.02
    ax.set_position(pos)

    pos = cbar.ax.get_position()
    pos.x0 += 0.02
    pos.x1 += 0.02
    pos.y0 += 0.04
    pos.y1 += 0.02
    cbar.ax.set_position(pos)

    fig.savefig(os.path.join(savedir, f'xmitt_{int(xs/1e3):03}.png'), dpi=300)
    plt.close(fig)
