"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import sonic_layer_depth, Config

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

cf = Config()

c_fields = np.load('data/processed/inputed_decomp.npz')

z_a = c_fields['z_a']
x_a = c_fields['x_a']

c_bg = c_fields['c_bg']
c_tilt = c_fields['c_tilt']
c_spice = c_fields['c_spice']
c_total = c_fields['c_total']

plt_i = z_a <= 150.
#c_field = c_bg[plt_i, :]
c_field = c_total[plt_i, :]

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)
sld_m = z_a[:, None] > sld_z
c_sld = np.ma.array(c_total, mask=sld_m)
mean_c = np.mean(c_sld, axis=0).data
sec_mean_c = mean_c.mean()

# mixed layer density def
sig_100 = c_fields['sig_total']
ml_depth_100 = z_a[np.argmax((sig_100 - sig_100[0, :] > 0.05), axis=0)]

fig, ax = plt.subplots(figsize=(cf.jasa_2clm,  3))
cm = ax.pcolormesh(x_a / 1e3, z_a[plt_i], c_field - 1500,
                   cmap=plt.cm.coolwarm,
                   vmax = sec_mean_c + 6 - 1500, vmin = sec_mean_c - 6 - 1500)

ax.text(1000, 163, "+ 1500.")

cb = fig.colorbar(cm)
cb.set_label('Sound speed (m/s)')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')

ax.set_ylim(150, 0)
ax.set_xlim(0, 970)

pos = ax.get_position()
pos.x1 += 0.14
pos.x0 -= 0.03
pos.y1 += 0.06
pos.y0 += 0.04
ax.set_position(pos)

pos = cb.ax.get_position()
pos.x1 += 0.12
pos.x0 += 0.12
pos.y1 += 0.06
pos.y0 += 0.04
cb.ax.set_position(pos)

#fig.savefig(join(savedir, 'sound_speed_transcet.png'), dpi=300)

ax.plot(x_a / 1e3, ml_depth_100, '#eb44e6', linewidth=1)
ax.plot(x_a / 1e3, sld_z, '0.2', linewidth=1)

fig.savefig(join(savedir, 'sound_speed_transcet_sld.png'), dpi=300)

#reg = linregress(x_a, sld_z)
#ax.plot(x_a / 1e3, x_a * reg.slope + reg.intercept, '0.4')

#fig.savefig(join(savedir, 'sound_speed_transcet_rgs.png'), dpi=300)
