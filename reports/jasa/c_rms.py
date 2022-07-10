import numpy as np
import matplotlib.pyplot as plt
from src import sonic_layer_depth, Config
from scipy.stats import linregress
from os.path import join

plt.style.use('elr')
cf = Config()

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

z_sld, sld_i = sonic_layer_depth(z_a, c_mean[:, None], z_max=300)
lrg = linregress(z_a[:sld_i[0]], c_mean[:sld_i[0]])

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(cf.jasa_1clm, 2.75))
ax[0].plot(c_mean, z_a, 'k')
ax[0].plot(z_a * lrg.slope + lrg.intercept, z_a, '--', color="b")
ax[0].plot([0, 1e4], [z_sld, z_sld], '0.4')
ax[0].text(1499.5, z_sld-7, 'SLD', bbox=bbox)

ax[1].plot(rms_total, z_a, label='observed')
ax[1].plot(rms_tilt, z_a, label='tilt')
ax[1].plot(rms_spice, z_a, label='spice')
ax[1].plot([0, 1e4], [z_sld, z_sld], '0.4')

ax[0].grid()
ax[1].grid()
ax[1].set_xlim(0, 2)

ax[1].legend(loc=3, bbox_to_anchor=(0.43, 0.45), handlelength=1, framealpha=1)
ax[0].set_ylim(150, 0)
ax[0].set_xlim(1498, 1512)

#ax[0].set_ylim(350, 0)
#ax[1].legend()
#ax[0].set_xlim(1490, 1512)

ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('Sound speed, $c$ (m/s)')
ax[1].set_xlabel('$c$ RMS (m/s)')
ax[1].set_xticks([0.5, 1, 1.5, 2])

pos = ax[0].get_position()
pos.x0 += 0.05
pos.x1 += -0.02
pos.y0 += 0.06
pos.y1 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += -0.06
pos.x1 += 0.06
pos.y0 += 0.06
pos.y1 += 0.06
ax[1].set_position(pos)

fig.savefig(join(savedir, 'rms_profile.png'), dpi=300)

print(lrg.slope)
