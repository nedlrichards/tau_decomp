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

from src import sonic_layer_depth, list_tl_files, Config

plt.ion()
cf = Config()

fc = 400
sec_num = 23
savedir = 'reports/jasa/figures'

tl_files = list_tl_files(fc, source_depth='shallow')
tl_data = np.load(tl_files[sec_num])

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


z_i = z_a < 160.

sld_z, _ = sonic_layer_depth(z_a[z_i], c_total[z_i, :])
sld_m = z_a[z_i, None] > sld_z
c_sld = np.ma.array(c_total[z_i, :], mask=sld_m)

c_plot_ref = np.mean(c_sld)

fig, axes = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(cf.jasa_2clm, 3.75))
vmax = np.ma.max(c_sld)
vmin = np.ma.min(c_sld)
x_t = x_a[0] / 1e3 + 2
cm = axes[0, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_tilt[z_i, :],
                            cmap=plt.cm.coolwarm,
                            vmax=vmax, vmin=vmin,
                            rasterized=True)
axes[0,0].text(x_t, 25, '(a)', bbox=cf.bbox, zorder=50, ha='center')
cm = axes[1, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_spice[z_i, :],
                            cmap=plt.cm.coolwarm,
                            vmax=vmax, vmin=vmin,
                            rasterized=True)
axes[1,0].text(x_t, 25, '(b)', bbox=cf.bbox, zorder=50, ha='center')
cm = axes[2, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_total[z_i, :],
                            cmap=plt.cm.coolwarm,
                            vmax=vmax, vmin=vmin,
                            rasterized=True)
axes[2,0].text(x_t, 25, '(c)', bbox=cf.bbox, zorder=50, ha='center')

axes[0, 0].set_ylim(150, 0)
axes[0, 0].set_xlim(x_a[0] / 1e3, x_a[-1] / 1e3)
yt = axes[0, 0].get_yticks()[1:]
axes[0, 0].set_yticks(yt[yt > 0])
#axes[0, 0].set_yticklabels(axes[0, 0].get_yticklabels())
#axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels()[0:])
#axes[2, 0].set_yticklabels(axes[2, 0].get_yticklabels()[0:])

cax = fig.add_axes([0.125, 0.91, 0.40, 0.03])
cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
cbar.set_label('Sound speed (m/s)')
cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
loc = plticker.MaxNLocator(nbins=4, integer=True)
cbar.ax.xaxis.set_major_locator(loc)
offset = matplotlib.transforms.ScaledTranslation(0, -0.05, fig.dpi_scale_trans)
for label in cbar.ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

cm = axes[0, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_tilt)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[1, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_spice)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[2, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_total)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

cax = fig.add_axes([0.565, 0.91, 0.40, 0.03])
cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
cbar.set_label('Acoustic pressure (dB re 1m)')
cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
cbar.set_ticks(cbar.get_ticks()[1:])
offset = matplotlib.transforms.ScaledTranslation(0, -0.05, fig.dpi_scale_trans)
for label in cbar.ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)


fig.supxlabel('Range (km)')
fig.supylabel('Depth (m)')
pos = axes[0, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
pos.y0 -= 0.02
pos.y1 -= 0.02
axes[0, 0].set_position(pos)

pos = axes[0, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
pos.y0 -= 0.02
pos.y1 -= 0.02
axes[0, 1].set_position(pos)

pos = axes[1, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
pos.y0 += 0.005
pos.y1 += 0.005
axes[1, 0].set_position(pos)

pos = axes[1, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
pos.y0 += 0.005
pos.y1 += 0.005
axes[1, 1].set_position(pos)

pos = axes[2, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
#pos.y0 += 0.07
pos.y0 += 0.03
pos.y1 += 0.03
axes[2, 0].set_position(pos)

pos = axes[2, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
#pos.y0 += 0.07
pos.y0 += 0.03
pos.y1 += 0.03
axes[2, 1].set_position(pos)


fig.savefig(os.path.join(savedir, 'decomp_xmission.png'), dpi=300)

