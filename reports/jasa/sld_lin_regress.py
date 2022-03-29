"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

sec4 = Section()
stab_height = sec4.stable_cntr_height(sec4.lvls)
#stab_spice = sec4.stable_spice(sec4.lvls)
stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_lvls = sec4.stable_spice(stab_height)

prof_i = 150

z_a = sec4.z_a

plt_i = z_a <= 150.
c_field = sec4.c[plt_i, :]

prop_i = sec4.z_a <= 150.

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(sec4.x_a / 1e3, sld_z, 'k')
reg = linregress(sec4.x_a, sld_z)
ax.plot(sec4.x_a / 1e3, sec4.x_a * reg.slope + reg.intercept, 'C0')
#ax[0].text(120, 20, f'm={reg.slope * 1e3:0.3f} m'+'  km$^{-1}$',
        #bbox=bbox)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Sonic layer depth (m)')

ax.grid()
ax.set_ylim(150, 0)
ax.set_xlim(0, 970)

pos = ax.get_position()
pos.x0 -= 0.01
pos.x1 += 0.08
pos.y0 += 0.04
pos.y1 += 0.08
ax.set_position(pos)

fig.savefig(join(savedir, 'sld_linregress.png'), dpi=300)

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6.5, 6))
ax[0].plot(sec4.x_a / 1e3, sld_z, 'k')
reg = linregress(sec4.x_a, sld_z)
ax[0].plot(sec4.x_a / 1e3, sec4.x_a * reg.slope + reg.intercept, 'C0')
#ax[0].text(120, 20, f'm={reg.slope * 1e3:0.3f} m'+'  km$^{-1}$',
        #bbox=bbox)

i_40 = np.argmin(np.abs(sec4.z_a - 40))
tau = sec4.spice[i_40, :]
reg = linregress(sec4.x_a, tau)
ax[1].plot(sec4.x_a / 1e3, tau, 'k')
ax[1].plot(sec4.x_a / 1e3, sec4.x_a * reg.slope + reg.intercept, 'C0')
#ax[1].text(20, 2.3, f'm={reg.slope * 1e3:0.3e}'+' kg/m$^3$  km$^{-1}$',
        #bbox=bbox)

sig = sec4.sigma0[i_40, :]
reg = linregress(sec4.x_a, sig)
ax[2].plot(sec4.x_a / 1e3, sig, 'k')
ax[2].plot(sec4.x_a / 1e3, sec4.x_a * reg.slope + reg.intercept, 'C0')
#ax[2].text(20, 24.95, f'm={reg.slope * 1e3:0.3e}' +'  kg /m$^3$ km$^{-1}$',
        #bbox=bbox)

ax[2].set_xlabel('Range (km)')
ax[0].set_ylabel('Sonic layer depth (m)')
ax[1].set_ylabel(r'$\tau$ (kg / m$^3$)')
ax[2].set_ylabel(r'$\sigma_0$ (kg / m$^3$)')

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].set_xlim(0, 970)
ax[0].set_ylim(150, 0)

pos = ax[0].get_position()
pos.x0 -= 0.01
pos.x1 += 0.08
pos.y0 += 0.04
pos.y1 += 0.10
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.01
pos.x1 += 0.08
pos.y0 += 0.005
pos.y1 += 0.065
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 -= 0.01
pos.x1 += 0.08
pos.y0 -= 0.03
pos.y1 += 0.03
ax[2].set_position(pos)

fig.savefig(join(savedir, 'sld_dens_linregress.png'), dpi=300)

