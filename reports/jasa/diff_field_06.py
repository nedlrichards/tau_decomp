"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.transforms
from copy import copy
import os

from src import sonic_layer_depth, list_tl_files, Config, section_cfield

plt.ion()
cf = Config()
savedir = 'reports/jasa/tex'

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_tilt = fields['c_tilt']
c_spice = fields['c_spice']
c_total = fields['c_total']

sld_bg, _ = sonic_layer_depth(z_a, c_bg, z_max=300)
sld_tilt, _ = sonic_layer_depth(z_a, c_tilt, z_max=300)
sld_spice, _ = sonic_layer_depth(z_a, c_spice, z_max=300)
sld_total, _ = sonic_layer_depth(z_a, c_total, z_max=300)

z_i = z_a < 350
fig, axes = plt.subplots(4, 1, figsize=(cf.jasa_2clm, 6.00), sharex=True)
vmax = np.ma.max(c_bg[z_i, :])
vmin = np.ma.min(c_bg[z_i, :])
cm0 = axes[0].pcolormesh(x_a / 1e3, z_a[z_i], c_bg[z_i, :] - 1500,
                            cmap=plt.cm.coolwarm,
                            vmax=1515 - 1500, vmin=1490 - 1500,
                            rasterized=True)
cm1 = axes[1].pcolormesh(x_a / 1e3, z_a[z_i], (c_tilt - c_bg)[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmax=4, vmin=-4,
                            rasterized=True)
cm = axes[2].pcolormesh(x_a / 1e3, z_a[z_i], (c_spice - c_bg)[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmax=4, vmin=-4,
                            rasterized=True)
cm = axes[3].pcolormesh(x_a / 1e3, z_a[z_i], (c_total - c_bg)[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmax=4, vmin=-4,
                            rasterized=True)

axes[0].plot(x_a / 1e3, sld_bg, linewidth=0.75, color='k')
axes[1].plot(x_a / 1e3, sld_tilt, linewidth=0.75, color='k')
axes[2].plot(x_a / 1e3, sld_spice, linewidth=0.75, color='k')
axes[3].plot(x_a / 1e3, sld_total, linewidth=0.75, color='k')

cax = fig.add_axes([0.92, 0.849, 0.01, 0.135])
fig.colorbar(cm0, cax=cax)

cax = fig.add_axes([0.92, 0.22, 0.01, 0.5])
fig.colorbar(cm1, cax=cax)

axes[0].text(985, 465, "+1500")
axes[0].text(985, 545, "$c$ (m/s)")
axes[3].text(985, 250, "$\delta \, c$ (m/s)")


axes[0].set_ylim(350, 0)
axes[1].set_ylim(350, 0)
axes[2].set_ylim(350, 0)
axes[3].set_ylim(350, 0)

axes[0].set_yticks([0, 175, 350])
axes[1].set_yticks([0, 175, 350])
axes[2].set_yticks([0, 175, 350])
axes[3].set_yticks([0, 175, 350])

#axes[1].set_yticklabels(['', 175, 350])
#axes[2].set_yticklabels(['', 175, 350])
#axes[3].set_yticklabels(['', 175, 350])

# Create offset transform by 5 points in x direction
dx = 0/72.
dy = -3/72.

offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
label = axes[1].yaxis.get_majorticklabels()[0]
label.set_transform(label.get_transform() + offset)
label = axes[2].yaxis.get_majorticklabels()[0]
label.set_transform(label.get_transform() + offset)
label = axes[3].yaxis.get_majorticklabels()[0]
label.set_transform(label.get_transform() + offset)

dy = 3/72.
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
label = axes[1].yaxis.get_majorticklabels()[-1]
label.set_transform(label.get_transform() + offset)
label = axes[2].yaxis.get_majorticklabels()[-1]
label.set_transform(label.get_transform() + offset)
label = axes[3].yaxis.get_majorticklabels()[-1]
label.set_transform(label.get_transform() + offset)


fig.supylabel('Depth (m)')
fig.supxlabel('Position, $x$ (km)')

dy1 = -0.03
y0 = 0.135
pos = axes[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y0 += y0
pos.y1 += y0 + dy1
axes[0].set_position(pos)

dy2 = 0.06
y0 = 0.09
pos = axes[1].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y0 += y0
pos.y1 += y0 + dy2
axes[1].set_position(pos)

y0 = 0.04
pos = axes[2].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y0 += y0
pos.y1 += y0 + dy2
axes[2].set_position(pos)

y0 = -0.015
pos = axes[3].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y0 += y0
pos.y1 += y0 + dy2
axes[3].set_position(pos)

axes[0].text(0, 0, 'bg', bbox=cf.bbox)
axes[1].text(0, 0, 'tilt', bbox=cf.bbox)
axes[2].text(0, 0, 'spice', bbox=cf.bbox)
axes[3].text(0, 0, 'total', bbox=cf.bbox)

fig.savefig(os.path.join(savedir, 'figure_6.pdf'), dpi=300)
"""
fig, ax = plt.subplots()
ax.plot(c_total[z_i, 670:680], z_a[z_i])
#ax.plot(c_total[z_i, 670:680], z_a[z_i])
ax.set_ylim(150, 0)

fig, ax = plt.subplots()
ax.plot((c_spice - c_bg)[z_i, 670:680], z_a[z_i])
#ax.plot(c_bg[z_i, 670:680], z_a[z_i])
ax.set_ylim(150, 0)

fig, ax = plt.subplots()
ax.plot((c_tilt - c_bg)[z_i, 670:680], z_a[z_i])
#ax.plot(c_bg[z_i, 670:680], z_a[z_i])
ax.set_ylim(150, 0)
"""
