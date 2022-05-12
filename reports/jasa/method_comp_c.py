from scipy.io import loadmat
import gsw
import numpy as np
import matplotlib.pyplot as plt

from src import Config, lvl_profiles, grid_field, Section, SA_CT_from_sigma0_spiciness0

plt.ion()
cf = Config()

# sound speed comparison
all_lvls = np.load('data/processed/inputed_decomp.npz')
z_a = all_lvls['z_a']
x_a = all_lvls['x_a']

press = gsw.p_from_z(-z_a, cf.lat)

# extrapolate from stable position into mixed layer
stable_lvls = all_lvls['stable_lvls']
filled_lvls = all_lvls['filled_lvls']
spice_lvls = np.concatenate([stable_lvls[[0], :, :], filled_lvls[[1], :, :]])
sig_rud, tau_rud = grid_field(z_a, spice_lvls, cf.sig_lvl[:spice_lvls.shape[1]])
sa_rud, ct_rud = SA_CT_from_sigma0_spiciness0(sig_rud, tau_rud)
c_rud = gsw.sound_speed(sa_rud, ct_rud, press[:, None])

# tau difference method
decomp_fields = np.load('data/processed/decomposed_fields.npz')
z_a_decomp = decomp_fields['z_a']
c_tau_d = decomp_fields['c_spice']
z_i = z_a_decomp < 300

c_diff = c_tau_d - c_rud

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.5))
cm = ax.pcolormesh(x_a / 1e3, z_a, c_diff, cmap=plt.cm.PuOr,
                      vmin=-2.5, vmax=2.5)
cb = fig.colorbar(cm)
cb.set_label('$\Delta /, c$ (m/s)')

ax.set_xlim(175, 400)
ax.set_ylim(150, 0)
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Position, $x$ (km)')

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.03
pos.y0 += 0.08
pos.y1 += 0.09
ax.set_position(pos)

pos = cb.ax.get_position()
pos.x0 += 0.03
pos.x1 += 0.03
pos.y0 += 0.08
pos.y1 += 0.09
cb.ax.set_position(pos)

fig.savefig('reports/jasa/figures/sound_speed_comp.png', dpi=300)

