import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gsw
from os.path import join

from src import Field, SectionLvls, Config

plt.ion()
cf = Config()
field = Field(spice_def=2)
sec4 = SectionLvls(spice_def=2)
sig_lvl = sec4.sig_lvl
mean_sa = sec4.stable_spice(sec4.lvls)[1]
mean_ct = gsw.CT_from_rho(sig_lvl[:, None] + 1e3, mean_sa, field.p_ref)[0]

position = np.broadcast_to(field.x_a, field.xy_ct.shape)
savedir = 'reports/jasa/figures'

# compute density contours
xbounds = [33.75, 34.9]
ybounds = [7.4, 18]
cntr_x = np.linspace(xbounds[0], xbounds[1], 100)
cntr_y = np.linspace(ybounds[0], ybounds[1], 100)
cntr_sig = gsw.sigma0(cntr_x[:, None], cntr_y[None, :])
cntr_c = gsw.sound_speed(cntr_x[:, None], cntr_y[None, :], 0)

norm = mpl.colors.Normalize(vmin=0, vmax=field.x_a[-1])

fig, ax = plt.subplots()
pth = ax.scatter(field.xy_sa.flatten(), field.xy_ct.flatten(), c=position.flatten(),
                 cmap=plt.cm.cividis, norm=norm, s=0.2, alpha=0.2, zorder=20)

cnt_lvls = np.array([24. , 24.4, 24.8, 25.2, 25.6, 26. , 26.4])
CS = ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2', linewidths=0.5, levels=cnt_lvls)
#CS = ax.contour(cntr_x, cntr_y, cntr_sig.T, colors='0.2')

cax = fig.add_axes([0.72, 0.20, 0.03, 0.2], fc='w', zorder=30)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.cividis),
                  cax=cax, ticks=[0, 500e3, 969e3])
cb.set_ticklabels(['0', '500', '970'])
ax.text(34.76, 7.9, '$x$ (km)', rotation='vertical', zorder=30)


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    #return rf"$\sigma$={s}"
    return rf"{s}"


ax.clabel(CS, CS.levels[::2], inline=True, fmt=fmt, fontsize=10)
ax.set_ylabel(r'Tempurature, $\theta$ ($^\circ$C)')
ax.set_xlabel(r'Salinity, S$_{A}$ (g kg$^{-1}$)')
ax.set_xlim(33.7, 34.9)
ax.set_ylim(7, 18)

pos = ax.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 += 0.06
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig(join(savedir, 'spice_ts.png'), dpi=300)

#ax.plot(mean_sa[:, 900], mean_ct[:, 900], color='r', zorder=40)

fig.savefig(join(savedir, 'spice_ts_mean.png'), dpi=300)

c_lvls = np.array([1490, 1500, 1510])
c_cnt = ax.contour(cntr_x, cntr_y, cntr_c.T, colors='0.4', linewidths=0.5, levels=c_lvls, alpha=0.8)
pos = [(33.8, 16.5), (33.8, 10.5)]
ax.clabel(c_cnt, c_lvls[[0, 2]], inline=True, fontsize=10, manual=pos)

patch = Rectangle((34.5, 7), 0.4, 3.8, alpha=1, color='w', ec='w', zorder=5)
ax.add_patch(patch)

fig.savefig(join(savedir, 'spice_ts_mean_c.png'), dpi=300)

1/0
fig, ax = plt.subplots()
pth = ax.scatter(field.xy_sig, field.xy_gamma, c=position, cmap=plt.cm.cividis, norm=norm, s=0.2, alpha=0.4)

cax = fig.add_axes([0.82, 0.20, 0.03, 0.2], fc='w', zorder=30)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.cividis),
                  cax=cax, ticks=[0, 400e3, 900e3])
cb.set_ticklabels(['0', '400', '900'])
ax.text(34.53, 9, 'x (km)')

ax.set_ylabel(r'spice, $\gamma$ (kg m$^{-3}$)')
ax.set_xlabel(r'density, $\sigma_0$ (kg m$^{-3}$)')
ax.text(26.02, -0.37, 'x (km)')
#ax.grid()

pos = ax.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 += 0.06
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig(join(savedir, 'spice_sig_gamma.png'), dpi=300)
