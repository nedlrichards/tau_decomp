"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/spice_po/figures'

sec4 = Section()
stab_height = sec4.stable_cntr_height(sec4.lvls)
#stab_spice = sec4.stable_spice(sec4.lvls)
stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_lvls = sec4.stable_spice(stab_height)

#z_a, c_total = sec4.compute_c_field(sec4.lvls)
#z_a, c_bg = sec4.compute_c_field(stab_lvls)
z_a = sec4.z_a

plt_i = z_a <= 150.
#c_field = c_bg[plt_i, :]
c_field = sec4.c[plt_i, :]

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)
sld_m = z_a[:, None] > sld_z
c_sld = np.ma.array(sec4.c, mask=sld_m)
mean_c = np.mean(c_sld, axis=0).data
sec_mean_c = mean_c.mean()

fig, ax = plt.subplots(figsize=(6.5, 3))
cm = ax.pcolormesh(sec4.x_a / 1e3, z_a[plt_i], c_field - 1500,
                   cmap=plt.cm.coolwarm,
                   vmax = sec_mean_c + 2 - 1500, vmin = sec_mean_c - 6 - 1500)

ax.text(310, 163, "+ 1500.")

cb = fig.colorbar(cm)
cb.set_label('Sound speed (m/s)')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')

ax.set_ylim(150, 0)
ax.set_xlim(0, 299)

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

fig.savefig(join(savedir, 'sound_speed_transcet.png'), dpi=300)

ax.plot(sec4.x_a / 1e3, sld_z, 'k')

fig.savefig(join(savedir, 'sound_speed_transcet_sld.png'), dpi=300)

