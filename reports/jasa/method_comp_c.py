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

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(cf.jasa_1clm, 4))
cm = ax[0].pcolormesh(x_a / 1e3, z_a, c_rud, cmap=plt.cm.coolwarm,
                      vmin=1497, vmax=1510)
cm = ax[1].pcolormesh(x_a / 1e3, z_a_decomp[z_i], c_tau_d[z_i, :],
                      cmap=plt.cm.coolwarm, vmin=1497, vmax=1510)

ax[0].set_xlim(0, 199)
ax[0].set_ylim(150, 0)
fig.supylabel('Depth (m)')
ax[1].set_xlabel('Position, $x$ (km)')

pos = ax[0].get_position()
pos.x0 += 0.07
pos.x1 += 0.07
pos.y0 += 0.04
pos.y1 += 0.09
ax[0].set_position(pos)


pos = ax[1].get_position()
pos.x0 += 0.07
pos.x1 += 0.07
pos.y0 += 0.02
pos.y1 += 0.07
ax[1].set_position(pos)

fig.savefig('reports/jasa/figures/sound_speed_comp.png', dpi=300)

