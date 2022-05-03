import numpy as np
import gsw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Config, lvl_profiles, grid_field, Section, SA_CT_from_sigma0_spiciness0

plt.ion()
cf = Config()
grid_data = loadmat('data/raw/section4_fields.mat')

x_a = np.squeeze(grid_data['xx'])
z_a = np.squeeze(grid_data['zz'])

press = gsw.p_from_z(-z_a, cf.lat)

# isopycnal contours of smooth field
total_sa = gsw.SA_from_SP(grid_data['sali'], press[:, None], cf.lon, cf.lat)
total_theta = grid_data['thetai']
total_sig = gsw.sigma0(total_sa, total_theta)

# grid salinity and theta into density bins
num_bins = 300
bins = np.linspace(total_sig.min(), total_sig.max(), num=num_bins)[1:]
props = []
for sig_prof, sa_prof, ct_prof in zip(total_sig.T, total_sa.T, total_theta.T):
    i_sa = np.interp(bins, sig_prof, sa_prof, left=np.nan, right=np.nan)
    i_ct = np.interp(bins, sig_prof, ct_prof, left=np.nan, right=np.nan)
    props.append(np.array([i_sa, i_ct]))
props = np.stack(props, axis=2)

mean_props = np.nanmean(props, axis=-1)

sa = props[0, :, :]
ct = props[1, :, :]
r = np.broadcast_to(x_a, ct.shape)

# interpolate mean profile
f_sig = total_sig.flatten()
mean_sa = np.interp(f_sig, bins, mean_props[0]).reshape(total_sig.shape)
mean_ct = np.interp(f_sig, bins, mean_props[1]).reshape(total_sig.shape)

alp_prof  = np.nanmean(gsw.alpha(total_sa, total_theta, press[:, None]))
beta_prof = np.nanmean(gsw.beta(total_sa, total_theta, press[:, None]))
r0 = 1000
diff_sa = beta_prof * r0 * (total_sa - mean_sa)
diff_ct = alp_prof * r0 * (total_theta - mean_ct)

gamma = np.sign(diff_ct) * np.sqrt(diff_ct ** 2 + diff_sa ** 2)
c = gsw.sound_speed(total_sa, total_theta, press[:, None])



# S-T plot
cmap = plt.cm.gray((0.6 * r / r.max()) + 0.2)
# set alpha value
cmap[:, :, -1] = 0.6

# compute density contours
xbounds = [33.75, 34.9]
ybounds = [7.4, 18]

cntr_x = np.linspace(xbounds[0], xbounds[1], 100)
cntr_y = np.linspace(ybounds[0], ybounds[1], 100)plot_i = 45
ax.plot(sec4.x_a / 1e3, spice[plot_i])
ax.plot(sec4.x_a / 1e3, np.sort(spice[plot_i])[::-1])
cntr_sig = gsw.sigma0(cntr_x[:, None], cntr_y[None, :])

#r_i = np.arange(sa.shape[-1])
#r_i = np.arange(500)
r_i = np.arange(500, sa.shape[-1])
#r_i = np.arange(136)
#r_i = np.arange(136, 500)
#r_i = np.arange(500, 747)
#r_i = np.arange(688, 747)
#r_i = np.arange(747, 870)
#r_i = np.arange(870, 916)
#r_i = np.arange(916, sa.shape[-1])

fig, ax = plt.subplots(figsize=(6,6))
sa_plot = sa[:, r_i].flatten()[::-1]
ct_plot = ct[:, r_i].flatten()[::-1]
r_plot = r[:, r_i].flatten()[::-1]
ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2')
pth = ax.scatter(sa_plot, ct_plot, c=r_plot,
                 cmap=plt.cm.gray_r,
                 vmin=-200, vmax=1200, zorder=10)
ax.plot(mean_props[0], mean_props[1], 'C1', linewidth=1, zorder=20)
ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel('$\Theta$ ($^o$C)')

cax = fig.add_axes([0.75, 0.15, 0.04, 0.4], fc='w', zorder=30)
cb = fig.colorbar(pth, cax=cax, ticks=[0, 200, 400, 600, 800, 1e3])
cb.patch.set_facecolor('w')

pos = ax.get_position()
pos.x0 += -0.02
pos.x1 += 0.06
pos.y0 += -0.02
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig('reports/figures/T_S.png', dpi=300)

#r_i = np.arange(sa.shape[-1])
#r_i = np.arange(500)
r_i = np.arange(500, sa.shape[-1])
#r_i = np.arange(136)
#r_i = np.arange(136, 500)
#r_i = np.arange(500, 747)
#r_i = np.arange(688, 747)
#r_i = np.arange(747, 870)
#r_i = np.arange(870, 916)
#r_i = np.arange(916, sa.shape[-1])

fig, ax = plt.subplots(figsize=(6,6))
r_i = np.arange(500)
sa_plot = sa[:, r_i].flatten()[::-1]
ct_plot = ct[:, r_i].flatten()[::-1]
r_plot = r[:, r_i].flatten()[::-1]
ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2')
pth = ax.scatter(sa_plot, ct_plot, c='C0', alpha=0.2,
                 cmap=plt.cm.gray_r,
                 vmin=-200, vmax=1200, zorder=10)

r_i = np.arange(500, sa.shape[-1])
sa_plot = sa[:, r_i].flatten()[::-1]
ct_plot = ct[:, r_i].flatten()[::-1]
r_plot = r[:, r_i].flatten()[::-1]
ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2')
pth = ax.scatter(sa_plot, ct_plot, c='C1', alpha=0.2,
                 cmap=plt.cm.gray_r,
                 vmin=-200, vmax=1200, zorder=10)
ax.plot(mean_props[0], mean_props[1], 'C1', linewidth=1, zorder=20)
ax.set_xlabel('Salinity (g/kg)')
ax.set_ylabel('$\Theta$ ($^o$C)')

#cax = fig.add_axes([0.75, 0.15, 0.04, 0.4], fc='w', zorder=30)
#cb = fig.colorbar(pth, cax=cax, ticks=[0, 200, 400, 600, 800, 1e3])
#cb.patch.set_facecolor('w')

pos = ax.get_position()
pos.x0 += -0.02
pos.x1 += 0.06
pos.y0 += -0.02
pos.y1 += 0.06
ax.set_position(pos)



fig, ax = plt.subplots(figsize=(7.5,4))
cm = ax.pcolormesh(x_a, z_a, gamma, cmap=plt.cm.coolwarm)
fig.colorbar(cm)
ax.set_ylim(200, 0)


fig, ax = plt.subplots(figsize=(6,6))
pth = ax.scatter(gamma.flatten(), f_sig, c=c - 1500,
                 vmin=1485 - 1500, vmax=1515 - 1500, alpha=0.4, zorder=10,
                 cmap=plt.cm.cividis, s=0.2)
cmap = pth.get_cmap()
sm = plt.cm.ScalarMappable(mpl.colors.Normalize(*pth.get_clim()), cmap)

ax.grid(zorder=1)
ax.set_ylim(26.6, 24.75)
ax.set_xlabel('$\gamma$ (kg/m$^3$)')
ax.set_ylabel('$\sigma$ (kg/m$^3$)')
cb = fig.colorbar(sm)
cb.set_label('Sound speed (m/s)')

ax.text(0.53, 26.65, '+ 1500.')

pos = ax.get_position()
pos.x0 += 0.02
pos.x1 += 0.06
pos.y0 += -0.02
pos.y1 += 0.06
ax.set_position(pos)

pos = cb.ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.06
pos.y0 += -0.02
pos.y1 += 0.06
cb.ax.set_position(pos)

fig.savefig('reports/figures/local_spice.png', dpi=300)
