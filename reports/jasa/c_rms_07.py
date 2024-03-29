import numpy as np
import matplotlib.pyplot as plt
from src import sonic_layer_depth, Config
from scipy.stats import linregress
from scipy.io import loadmat
from os.path import join

plt.style.use('elr')
cf = Config()

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/tex'

c_fields = np.load('data/processed/inputed_decomp.npz')
gm = loadmat('data/external/GMStats_spice.mat')

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
rms_gm = np.interp(z_a, gm['zz'][0], gm['rmsdcGM'][0])

z_sld, sld_i = sonic_layer_depth(z_a, c_mean[:, None], z_max=300)
lrg = linregress(z_a[:sld_i[0]], c_mean[:sld_i[0]])

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(cf.jasa_1clm, 2.75))
ax[0].plot(c_mean, z_a, 'k')
ax[0].plot(z_a * lrg.slope + lrg.intercept, z_a, '--', color="b")
ax[0].plot([0, 1e4], [z_sld, z_sld], '0.4')
#ax[0].text(1492.5, z_sld-19, 'SLD', bbox=bbox)

l0, = ax[1].plot(rms_total, z_a, label='observed')
l1, = ax[1].plot(rms_tilt, z_a, label='tilt')
l2, = ax[1].plot(rms_spice, z_a, label='spice')
l3, = ax[1].plot(rms_gm, z_a, label='GM', linestyle=(0, (10, 3)), color='xkcd:kelly green')
ax[1].plot([0, 1e4], [z_sld, z_sld], '0.4')

ax[0].grid()
ax[1].grid()
ax[1].set_xlim(0, 2)

#ax[1].legend(loc=3, bbox_to_anchor=(0.37, 0.05), handlelength=1, framealpha=1)
l3, = ax[1].plot([], linestyle='--', label='GM', color='xkcd:kelly green')
ax[1].legend(handlelength=1.2, framealpha=1, handles=[l0, l1, l2, l3])
ax[0].set_ylim(350, 0)
ax[0].set_xlim(1490, 1512)

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

fig.savefig(join(savedir, 'figure_7.pdf'), dpi=300)

print(lrg.slope)
