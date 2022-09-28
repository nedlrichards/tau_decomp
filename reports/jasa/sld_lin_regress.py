"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.io import loadmat
from os.path import join

from src import sonic_layer_depth, lvl_profiles, Config
import gsw

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

c_fields = np.load('data/processed/inputed_decomp.npz')

z_a = c_fields['z_a']
x_a = c_fields['x_a']

c_bg = c_fields['c_bg']
c_tilt = c_fields['c_tilt']
c_spice = c_fields['c_spice']
c_total = c_fields['c_total']

prof_i = 150

plt_i = z_a <= 150.
c_field = c_total[plt_i, :]

prop_i = z_a <= 150.

sld_z, _ = sonic_layer_depth(z_a[plt_i], c_field)

# because I used sigma referenced to 100 m, 25.4 contour is not in same place
grid_data = loadmat('data/processed/stablized_field.mat')
cf = Config()
x_a = np.squeeze(grid_data['x_a']).astype(np.float64) * 1e3
z_a = np.squeeze(grid_data['z_a']).astype(np.float64)
press = gsw.p_from_z(-z_a, cf.lat)
# load stabalized properties
xy_sa = (grid_data['SA_stable'].T).astype(np.float64)
xy_ct = (grid_data['CT_stable'].T).astype(np.float64)
xy_sig = gsw.sigma0(xy_sa, xy_ct)

sig_lvl = lvl_profiles(z_a, xy_sig, np.zeros_like(xy_sig), [25.4])

# mixed layer density def
ml_depth = z_a[np.argmax((xy_sig - xy_sig[0, :] > 0.05), axis=0)]


fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(x_a / 1e3, sld_z, 'k')
ax.plot(x_a / 1e3, sig_lvl[0, 0, :], 'C3')
ax.plot(x_a / 1e3, ml_depth, 'C4')
reg = linregress(x_a, sld_z)
#ax.plot(x_a / 1e3, x_a * reg.slope + reg.intercept, 'C0')
#ax[0].text(120, 20, f'm={reg.slope * 1e3:0.3f} m'+'  km$^{-1}$',
        #bbox=bbox)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Sonic layer depth (m)')

ax.grid()
ax.set_ylim(200, 0)
ax.set_xlim(0, 970)

pos = ax.get_position()
pos.x0 -= 0.01
pos.x1 += 0.08
pos.y0 += 0.04
pos.y1 += 0.08
ax.set_position(pos)

#fig.savefig(join(savedir, 'sld_linregress.png'), dpi=300)
"""
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

"""
