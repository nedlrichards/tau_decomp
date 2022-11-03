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

fields = np.load('data/processed/inputed_decomp.npz')
_, c_bg = section_cfield(tl_data['xs'], fields['x_a'], fields['c_bg'])
_, c_tilt = section_cfield(tl_data['xs'], fields['x_a'], fields['c_tilt'])
_, c_spice = section_cfield(tl_data['xs'], fields['x_a'], fields['c_spice'])
_, c_total = section_cfield(tl_data['xs'], fields['x_a'], fields['c_total'])

z_i = z_a < 360.

sld_z_tilt, _ = sonic_layer_depth(z_a[z_i], c_tilt[z_i, :])
sld_z_spice, _ = sonic_layer_depth(z_a[z_i], c_spice[z_i, :])
sld_z_total, _ = sonic_layer_depth(z_a[z_i], c_total[z_i, :])

c_tilt -= c_bg
c_spice -= c_bg
c_total -= c_bg


sld_z, _ = sonic_layer_depth(z_a[z_i], c_bg[z_i, :])
sld_m = z_a[z_i, None] > sld_z
c_sld = np.ma.array(c_bg[z_i, :], mask=sld_m)

c_plot_ref = np.mean(c_sld)

fig, axes = plt.subplots(4, 2, figsize=(cf.jasa_2clm, 5.50))
vmax = np.ma.max(c_sld)
vmin = np.ma.min(c_sld)
x_t = x_a[0] / 1e3 - 2

cm0 = axes[0, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_bg[z_i, :],
                            cmap=plt.cm.coolwarm,
                            vmax=1507, vmin=1510,
                            rasterized=True)
axes[0,0].text(x_t, 30, '(a)', bbox=cf.bbox, zorder=50, ha='center')
cm1 = axes[1, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_tilt[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmin=-3, vmax=3,
                            rasterized=True)
axes[1,0].text(x_t, 30, '(b)', bbox=cf.bbox, zorder=50, ha='center')
cm = axes[2, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_spice[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmin=-3, vmax=3,
                            rasterized=True)
axes[2,0].text(x_t, 30, '(c)', bbox=cf.bbox, zorder=50, ha='center')
cm = axes[3, 0].pcolormesh(x_a / 1e3, z_a[z_i], c_total[z_i, :],
                            cmap=plt.cm.BrBG,
                            vmin=-3, vmax=3,
                            rasterized=True)
axes[3,0].text(x_t, 30, '(d)', bbox=cf.bbox, zorder=50, ha='center')

axes[1, 0].plot(x_a / 1e3, sld_z_tilt, linewidth=2, color='k')
axes[2, 0].plot(x_a / 1e3, sld_z_spice, linewidth=2, color='k')
axes[3, 0].plot(x_a / 1e3, sld_z_total, linewidth=2, color='k')

yt = axes[0, 0].get_yticks()[1:]
#axes[0, 0].set_yticks(yt[yt > 0])
#axes[0, 0].set_yticklabels(axes[0, 0].get_yticklabels())
#axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels()[0:])
#axes[2, 0].set_yticklabels(axes[2, 0].get_yticklabels()[0:])

cax = fig.add_axes([0.200, 0.93, 0.25, 0.015])
cbar = fig.colorbar(cm0, cax=cax, orientation='horizontal')
cax.text(-0.30, 0, '$c$ (m/s)', transform=cax.transAxes)
cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
cbar.set_ticks([1507, 1508.5, 1510])
cbar.set_ticklabels(['1507', '1508.5', '1510'])
offset = matplotlib.transforms.ScaledTranslation(0, -0.05, fig.dpi_scale_trans)
for i, label in enumerate(cbar.ax.xaxis.get_majorticklabels()):
    if i == 0:
        label.set_transform(label.get_transform() +
            matplotlib.transforms.ScaledTranslation(0.15, -0.05, fig.dpi_scale_trans))
    else:
        label.set_transform(label.get_transform() + offset)


cax = fig.add_axes([0.200, 0.70, 0.25, 0.015])
cbar = fig.colorbar(cm1, cax=cax, orientation='horizontal')
cax.text(-0.35, 0, '$\delta c$ (m/s)', transform=cax.transAxes)
cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
cbar.set_ticks([-3, 0, 3])

offset = matplotlib.transforms.ScaledTranslation(0, -0.05, fig.dpi_scale_trans)
for i, label in enumerate(cbar.ax.xaxis.get_majorticklabels()):
    if i == 0:
        label.set_transform(label.get_transform() +
            matplotlib.transforms.ScaledTranslation(0.10, -0.05, fig.dpi_scale_trans))
    else:
        label.set_transform(label.get_transform() + offset)

cm = axes[0, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_bg)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[1, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_tilt)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[2, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_spice)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[3, 1].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_total)).T,
                        cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

cax = fig.add_axes([0.680, 0.93, 0.25, 0.015])
cbar = fig.colorbar(cm, cax=cax, orientation='horizontal')
#cbar.set_label('$p$ (dB re 1 m)', loc='left')
#cax.text(-4.8, -2.2, '$p$ (dB re 1 m)')
cax.text(-0.56, 0, '$p$ (dB re 1 m)', transform=cax.transAxes)

cbar.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
cbar.set_ticks(cbar.get_ticks()[1:])
offset = matplotlib.transforms.ScaledTranslation(0, -0.05, fig.dpi_scale_trans)
for label in cbar.ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

for ax in axes:
    for a in ax:
        a.set_ylim(350, 0.01)
        a.set_xlim(x_a[0] / 1e3, x_a[-1] / 1e3 - 1)

axes[0,0].set_xticks([])
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])
axes[1,0].set_xticks([])
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])
axes[2,0].set_xticks([])
axes[2,1].set_xticks([])
axes[2,1].set_yticks([])
axes[3,1].set_yticks([])

fig.supxlabel('Range (km)')
fig.supylabel('Depth (m)')

dy0 = -0.03
yoff = 0.05
pos = axes[0, 0].get_position()
pos.x0 += -0.02
pos.x1 += 0.06
pos.y0 += yoff
pos.y1 += yoff + dy0
axes[0, 0].set_position(pos)

pos = axes[0, 1].get_position()
pos.x0 += 0.00
pos.x1 += 0.08
pos.y0 += yoff
pos.y1 += yoff + dy0
axes[0, 1].set_position(pos)

dy1 = 0.01
yoff = -0.02
pos = axes[1, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[1, 0].set_position(pos)

pos = axes[1, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[1, 1].set_position(pos)

yoff = -0.02
pos = axes[2, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[2, 0].set_position(pos)

pos = axes[2, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[2, 1].set_position(pos)

yoff = -0.02
pos = axes[3, 0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.06
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[3, 0].set_position(pos)

pos = axes[3, 1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.08
pos.y0 += yoff
pos.y1 += yoff + dy1
axes[3, 1].set_position(pos)


fig.savefig(os.path.join(savedir, 'decomp_xmission.png'), dpi=300)

