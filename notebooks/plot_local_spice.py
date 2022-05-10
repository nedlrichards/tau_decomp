import numpy as np
import gsw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Config, lvl_profiles, grid_field, Section, SA_CT_from_sigma0_spiciness0
from src import LocalSpice

plt.ion()
cf = Config()
spice = LocalSpice()

# sorted spice along an isopycnal
max_x = 500  # stop at 500 km front
x_i = spice.x_a < max_x
pre_x_a = spice.x_a[x_i]

def sort_iso(iso_ind):
    iso_sig = spice.xsig_spice[iso_ind]
    pre_front = iso_sig[x_i]
    pre_sort = np.full_like(pre_front, np.nan)
    pre_sort[~np.isnan(pre_front)] = np.sort(pre_front[~np.isnan(pre_front)])
    return pre_front, pre_sort

fig, ax = plt.subplots(figsize=(7.5, 4))
p_i = 60
pre_front, pre_sort = sort_iso(p_i)
ax.text(80, pre_front[~np.isnan(pre_front)][0],
        f'$\sigma$={spice.sig_bins[p_i]:.2f}', bbox=cf.bbox)
ax.plot(pre_x_a, pre_front, 'C0', linewidth=1)
ax.plot(pre_x_a, pre_sort, 'k', linewidth=1)

p_i = 63
pre_front, pre_sort = sort_iso(p_i)
ax.text(80, pre_front[~np.isnan(pre_front)][0],
        f'$\sigma$={spice.sig_bins[p_i]:.2f}', bbox=cf.bbox)

ax.plot(pre_x_a, pre_front, 'C0', linewidth=1)
ax.plot(pre_x_a, pre_sort, 'k', linewidth=1)

p_i = 66
pre_front, pre_sort = sort_iso(p_i)
ax.text(80, pre_front[~np.isnan(pre_front)][0],
        f'$\sigma$={spice.sig_bins[p_i]:.2f}', bbox=cf.bbox)
pre_front, pre_sort = sort_iso(p_i)
ax.plot(pre_x_a, pre_front, 'C0', linewidth=1)
ax.plot(pre_x_a, pre_sort, 'k', linewidth=1)

ax.set_xlabel('Range (km)')
ax.set_ylabel('$\gamma$ (kg/m$^3$)')

ax.set_xlim(75, 510)

pos = ax.get_position()
pos.x0 -= 0.02
pos.x1 += 0.08
pos.y0 += 0.02
pos.y1 += 0.06
ax.set_position(pos)


# S-T plot
cmap = plt.cm.gray((0.6 * spice.x_a / spice.x_a.max()) + 0.2)
# set alpha value
cmap[:, -1] = 0.6

# compute density contours
xbounds = [33.75, 34.9]
ybounds = [7.4, 18]
cntr_x = np.linspace(xbounds[0], xbounds[1], 100)
cntr_y = np.linspace(ybounds[0], ybounds[1], 100)
cntr_sig = gsw.sigma0(cntr_x[:, None], cntr_y[None, :])

r_i = np.arange(spice.xy_sa.shape[-1])
#r_i = np.arange(500)
#r_i = np.arange(500, sa.shape[-1])
#r_i = np.arange(136)
#r_i = np.arange(136, 500)
#r_i = np.arange(500, 747)
#r_i = np.arange(688, 747)
#r_i = np.arange(747, 870)
#r_i = np.arange(870, 916)
#r_i = np.arange(916, sa.shape[-1])

fig, ax = plt.subplots(figsize=(6,6))
sa_plot = spice.xy_sa[:, r_i].flatten()[::-1]
ct_plot = spice.xy_ct[:, r_i].flatten()[::-1]
r_plot = spice.r_prof[:, r_i].flatten()[::-1]
ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2')
pth = ax.scatter(sa_plot, ct_plot, c=r_plot,
                 cmap=plt.cm.gray_r,
                 vmin=-200, vmax=1200, zorder=10)
ax.plot(spice.sig_mean_sa, spice.sig_mean_ct, 'C1', linewidth=1, zorder=20)
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

#fig.savefig('reports/figures/T_S.png', dpi=300)

z_max = 150.
z_i = spice.z_a < z_max
fig, ax = plt.subplots(figsize=(7.5,4))
cm = ax.pcolormesh(spice.x_a, spice.z_a[z_i],
                   spice.xy_spice[z_i, :], cmap=plt.cm.coolwarm)
fig.colorbar(cm)
ax.set_ylim(z_max, 0)

fig, ax = plt.subplots(figsize=(6,6))
pth = ax.scatter(spice.xy_spice.flatten(), spice.xy_sig.flatten(),
                 c=spice.xy_c - 1500, vmin=1485 - 1500, vmax=1515 - 1500,
                 alpha=0.4, zorder=10, cmap=plt.cm.cividis, s=0.2)
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
