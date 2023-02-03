"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join
import matplotlib.ticker as plticker
import matplotlib.transforms

import gsw

from src import sonic_layer_depth, Config, Field, SectionLvls

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

cf = Config()

field = Field(spice_def=2)
sec4 = SectionLvls(spice_def=2)
sig_lvl = sec4.sig_lvl
mean_sa = sec4.stable_spice(sec4.lvls)[1]
mean_ct = gsw.CT_from_rho(sig_lvl[:, None] + 1e3, mean_sa, field.p_ref)[0]

#prof_i = np.array([33, 249])
prof_i = np.array([249, 252])
#prof_i = np.array([20, 33, 40])
#prof_i = np.array([139])

z_a = field.z_a

plt_i = z_a <= 350.
c_xy = field.xy_c
sa_xy = field.xy_sa
ct_xy = field.xy_ct
sig_xy = field.xy_sig

sld_z, _ = sonic_layer_depth(z_a, c_xy)
ml_d = z_a[np.argmax((sig_xy[:, prof_i] - sig_xy[0, prof_i]) > 0.05, axis=0)]
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(cf.jasa_2clm, 3))

ax[0].plot(c_xy[:, prof_i], z_a)
ax_t0 = plt.twiny(ax=ax[0])
ax_t0.plot(sig_xy[:, prof_i], z_a, '--')

ax_t1 = plt.twiny(ax=ax[1])

# subtract mean from salinity and temperature
for p_i in prof_i:
    sa_ref = np.interp(sig_xy[:, p_i], sig_lvl, mean_sa[:, p_i])
    #ax[1].plot(sa_xy[:, p_i] - sa_ref, z_a)
    ax[1].plot(sa_xy[:, p_i], z_a)

    ct_ref = np.interp(sig_xy[:, p_i], sig_lvl, mean_ct[:, p_i])
    #ax_t1.plot(ct_xy[:, p_i] - ct_ref, z_a, '--')
    ax_t1.plot(ct_xy[:, p_i], z_a, '--')


#[a.set_prop_cycle(None) for a in ax]
#ax[0].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
#ax[1].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
#ax[0].plot([-10, 1e4], [ml_d[0], ml_d[0]], color='#eb44e6')
ax[0].plot([1506, 1e4], [sld_z[prof_i], sld_z[prof_i]], color='0.2')

ax[0].set_ylabel('Depth (m)')
ax[1].set_xlabel(r'Salinity, $S$ (kg/m$^3$)')
ax_t0.set_xlabel('Density, $\sigma_0$ (kg/m$^3$)')
ax[0].set_xlabel('Sound speed, c (m/s)')
ax_t1.set_xlabel(r'Temperature, $\theta$ ($^\circ$C)')

ax[0].text(1487, -30, '(a)', bbox=cf.bbox, clip_on=False)
ax[1].text(34.2, -30, '(b)', bbox=cf.bbox, clip_on=False)
ax[0].text(1503, 73, 'SLD', bbox=cf.bbox, clip_on=False)

ax[0].plot([0.12, 0.20], [1.17, 1.17], color='0.2', clip_on=False, transform=ax[0].transAxes)
ax[1].plot([0.12, 0.20], [1.17, 1.17], color='0.2', clip_on=False, transform=ax[1].transAxes)
ax[0].plot([0.12, 0.20], [-0.17, -0.17], '--', color='0.2', clip_on=False, transform=ax[0].transAxes)
ax[1].plot([0.12, 0.20], [-0.19, -0.19], '--', color='0.2', clip_on=False, transform=ax[1].transAxes)

# Create offset transform by 5 points in x direction
dx = 10/72.
dy = 0/72.

#offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
#label = ax[1].xaxis.get_majorticklabels()[0]
#label.set_transform(label.get_transform() + offset)

#label = ax[2].xaxis.get_majorticklabels()[0]
#label.set_transform(label.get_transform() + offset)

#ax[0].grid()
#ax[1].grid()

#ax[0].legend([s + ' km' for s in map(str, list(prof_i))], loc=(0.03, 0.78))
ax[0].legend([s + ' km' for s in map(str, list(prof_i))], loc=(0.3, 0.10))

ax_t0.set_xlim(25.1, 26.4)
ax[1].set_xlim(34.25, 34.8)
ax[0].set_xlim(1490, 1512.5)
ax[0].set_ylim(350, 0)

pos = ax[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y1 += -0.04
pos.y0 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.02
pos.y1 += -0.04
pos.y0 += 0.06
ax[1].set_position(pos)

fig.savefig(join(savedir, 'sld_profile.png'), dpi=300)

"""
# compute gradient in mixed layer
z_i = z_a[plt_i] < sld_z[prof_i[-1]]
y = c_xy[z_i, :][:, prof_i[-1]]
fit = linregress(z_a[plt_i][z_i], y=y)
print(fit.slope)
#ax[2].plot(fit.intercept + z_a * fit.slope - 1500., z_a, color='0.4', alpha=0.5)

z_i = z_a[plt_i] < sld_z[prof_i[0]]
y = c_xy[z_i, :][:, prof_i[0]]
fit = linregress(z_a[plt_i][z_i], y=y)
print(fit.slope)
#ax[2].plot(fit.intercept + z_a * fit.slope - 1500., z_a, color='0.4', alpha=0.5)


fig.savefig(join(savedir, 'sld_profile_grad.png'), dpi=300)
"""
