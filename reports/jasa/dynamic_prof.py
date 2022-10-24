"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join
import matplotlib.ticker as plticker
import matplotlib.transforms

from src import sonic_layer_depth, Config

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

cf = Config()

inputed_data  = np.load('data/processed/inputed_decomp.npz')

#prof_i = np.arange(250, 255)
prof_i = np.array([249, 252])
#prof_i = np.array([20, 33, 40])
#prof_i = np.array([139])

z_a = inputed_data['z_a']
z_a_sub = inputed_data['z_a_sub']

plt_i = z_a <= 350.
c_field = inputed_data['c_total'][plt_i, :]
spice_field = inputed_data['tau_total']
sig_field = inputed_data['sig_total']

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)
ml_d = z_a[np.argmax((sig_field[:, prof_i] - sig_field[0, prof_i]) > 0.05, axis=0)]
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(cf.jasa_2clm, 3))

ax[0].plot(sig_field[:, prof_i], z_a_sub)
ax[1].plot(spice_field[:, prof_i], z_a_sub)
ax[2].plot(c_field[:, prof_i], z_a[plt_i])

#[a.set_prop_cycle(None) for a in ax]
#ax[0].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
#ax[1].plot([-10, 100], [sld_z[prof_i], sld_z[prof_i]], alpha=0.6)
ax[0].plot([-10, 1e4], [ml_d[0], ml_d[0]], color='#eb44e6')
ax[2].plot([-10, 1e4], [sld_z[prof_i], sld_z[prof_i]], color='0.2')

ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('$\sigma_0$ (kg/m$^3$)')
ax[1].set_xlabel(r'$\gamma$ (kg/m$^3$)')
ax[2].set_xlabel('c (m/s)')

ax[0].text(25.37, 7, '(a)', bbox=cf.bbox)
ax[1].text(0.012, 7, '(b)', bbox=cf.bbox)
ax[2].text(1491, 7.0, '(c)', bbox=cf.bbox)

# Create offset transform by 5 points in x direction
dx = 10/72.
dy = 0/72.

offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

# apply offset transform to all x ticklabels.
label = ax[1].xaxis.get_majorticklabels()[0]
label.set_transform(label.get_transform() + offset)

label = ax[2].xaxis.get_majorticklabels()[0]
label.set_transform(label.get_transform() + offset)

ax[0].grid()
ax[1].grid()
ax[2].grid()

#ax[0].legend([s + ' km' for s in map(str, list(prof_i))], loc=(0.03, 0.78))
ax[0].legend([s + ' km' for s in map(str, list(prof_i))])

ax[0].set_xlim(25.3, 27.0)
loc = plticker.MultipleLocator(base=0.5)
ax[0].xaxis.set_major_locator(loc)
ax[1].set_xlim(-0., 0.3)
ax[2].set_xlim(1490, 1512.5)
ax[2].set_ylim(350, 0)

pos = ax[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.00
pos.y1 += 0.07
pos.y0 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.00
pos.x1 += 0.02
pos.y1 += 0.07
pos.y0 += 0.06
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.02
pos.x1 += 0.04
pos.y1 += 0.07
pos.y0 += 0.06
ax[2].set_position(pos)

fig.savefig(join(savedir, 'sld_profile.png'), dpi=300)

# compute gradient in mixed layer
z_i = z_a[plt_i] < sld_z[prof_i[-1]]
y = c_field[z_i, :][:, prof_i[-1]]
fit = linregress(z_a[plt_i][z_i], y=y)
print(fit.slope)
#ax[2].plot(fit.intercept + z_a * fit.slope - 1500., z_a, color='0.4', alpha=0.5)

z_i = z_a[plt_i] < sld_z[prof_i[0]]
y = c_field[z_i, :][:, prof_i[0]]
fit = linregress(z_a[plt_i][z_i], y=y)
print(fit.slope)
#ax[2].plot(fit.intercept + z_a * fit.slope - 1500., z_a, color='0.4', alpha=0.5)


fig.savefig(join(savedir, 'sld_profile_grad.png'), dpi=300)
