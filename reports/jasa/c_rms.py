import numpy as np
import matplotlib.pyplot as plt
from src import sonic_layer_depth
from os.path import join

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

c_fields = np.load('data/processed/inputed_decomp.npz')

z_a = c_fields['z_a']
x_a = c_fields['x_a']

c_bg = c_fields['c_bg']
c_tilt = c_fields['c_tilt']
c_spice = c_fields['c_spice']
c_total = c_fields['c_total']

c_mean = np.mean(c_total, axis=1)
rms_tilt = np.sqrt(np.var(c_tilt - c_bg, axis=1))
rms_spice = np.sqrt(np.var(c_spice - c_bg, axis=1))
rms_total = np.sqrt(np.var(c_total - c_bg, axis=1))

z_sld, _ = sonic_layer_depth(z_a, c_mean[:, None], z_max=300)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(4, 3))
ax[0].plot(c_mean, z_a, 'k')
ax[0].plot([0, 1e4], [z_sld, z_sld], '0.4')
ax[0].text(1498.5, z_sld-7, 'SLD', bbox=bbox)

ax[1].plot(rms_total, z_a, 'k', label='measured')
ax[1].plot(rms_tilt, z_a, 'C0', label='tilt')
ax[1].plot(rms_spice, z_a, 'C1', label='spice')
ax[1].plot([0, 1e4], [z_sld, z_sld], '0.4')

ax[0].grid()
ax[1].grid()
ax[1].set_xlim(0, 2)

ax[1].legend(loc=3, bbox_to_anchor=(0.48, 0.60))
ax[0].set_ylim(150, 0)
ax[0].set_xlim(1498, 1512)

#ax[0].set_ylim(350, 0)
#ax[1].legend()
#ax[0].set_xlim(1490, 1512)

fig.supylabel('Depth (m)')
ax[0].set_xlabel('Sound speed (m/s)')
ax[1].set_xlabel('RMS sound speed (m/s)')
ax[1].set_xticks([0.5, 1, 1.5, 2])

pos = ax[0].get_position()
pos.x0 += 0.04
pos.x1 -= 0.04
pos.y0 += 0.05
pos.y1 += 0.07
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.08
pos.x1 += 0.04
pos.y0 += 0.05
pos.y1 += 0.07
ax[1].set_position(pos)

fig.savefig(join(savedir, 'rms_profile.png'), dpi=300)

"""
plt_z_i = z_a <= 150.
plt_x_i = x_a <= 400e3

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
cm = ax[0].pcolormesh(x_a[plt_x_i] / 1e3, z_a[plt_z_i],
                      (c_total - c_bg)[np.ix_(plt_z_i, plt_x_i)],
                      vmin=-3, vmax=3, cmap=plt.cm.coolwarm)
cm = ax[1].pcolormesh(x_a[plt_x_i] / 1e3, z_a[plt_z_i],
                      (c_tilt - c_bg)[np.ix_(plt_z_i, plt_x_i)],
                      vmin=-3, vmax=3, cmap=plt.cm.coolwarm)
cm = ax[2].pcolormesh(x_a[plt_x_i] / 1e3, z_a[plt_z_i],
                      (c_spice - c_bg)[np.ix_(plt_z_i, plt_x_i)],
                      vmin=-3, vmax=3, cmap=plt.cm.coolwarm)

ax[0].set_ylim(150, 0)
"""
