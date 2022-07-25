from scipy.io import loadmat
import gsw
import numpy as np
import matplotlib.pyplot as plt

from src import Config

plt.ion()
cf = Config()

# sound speed comparison
decomp_fields = np.load('data/processed/inputed_decomp.npz')
z_a = decomp_fields['z_a']
z_a_sub = decomp_fields['z_a_sub']
x_a = decomp_fields['x_a']

z_i = z_a < 300

c_spice = decomp_fields['c_spice'][z_i, :]
c_rud = decomp_fields['c_spice_dmr'][z_i, :]

c_diff = c_spice - c_rud

# plot where methods switch
stable_lvls = decomp_fields['stable_lvls']
filled_lvls = decomp_fields['filled_lvls']

last_cntr_i = np.argmax(stable_lvls[0, :, :] > 1, axis=0)

last_z_hp = filled_lvls[0, last_cntr_i, np.arange(stable_lvls.shape[-1])]
last_z_hp_i = np.argmin(np.abs(last_z_hp - z_a[:, None]), axis=0)

last_z = stable_lvls[0, last_cntr_i, np.arange(stable_lvls.shape[-1])]
last_z_i = np.argmin(np.abs(last_z - z_a[:, None]), axis=0)


fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.25))
cm = ax.pcolormesh(x_a / 1e3, z_a[z_i], c_diff, cmap=plt.cm.BrBG,
                      vmin=-2.5, vmax=2.5)
ax.plot(x_a / 1e3, last_z, 'k')
ax.plot(x_a / 1e3, last_z_hp, '0.4', linewidth=0.5)
#cm = ax.pcolormesh(x_a / 1e3, z_a[z_i], c_rud)

cb = fig.colorbar(cm)
cb.set_label('$\Delta \, c$ (m/s)')

#ax.set_xlim(175, 275)
ax.set_xlim(230, 290)
ax.set_ylim(125, 0)
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Position, $x$ (km)')

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.03
pos.y0 += 0.10
pos.y1 += 0.09
ax.set_position(pos)

pos = cb.ax.get_position()
pos.x0 += 0.03
pos.x1 += 0.03
pos.y0 += 0.10
pos.y1 += 0.09
cb.ax.set_position(pos)

fig.savefig('reports/jasa/figures/sound_speed_comp.png', dpi=300)
