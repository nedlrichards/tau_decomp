from scipy.io import loadmat
import gsw
import numpy as np
import matplotlib.pyplot as plt

from src import Config, lvl_profiles, grid_field, Section, SA_CT_from_sigma0_spiciness0

plt.ion()
cf = Config()
rud_meth = loadmat('data/raw/section4_fields.mat')

x_a = np.squeeze(rud_meth['xx'])
z_a = np.squeeze(rud_meth['zz'])

press = gsw.p_from_z(-z_a, cf.lat)

# isopycnal contours of smooth field
smooth_sa = gsw.SA_from_SP(rud_meth['ssmooth'], press[:, None], cf.lon, cf.lat)
smooth_sig = gsw.sigma0(smooth_sa, rud_meth['thsmooth'])
smooth_lvls = lvl_profiles(z_a, smooth_sig, rud_meth['thsmooth'], cf.sig_lvl)

# isopycnal contours of total field
total_sa = gsw.SA_from_SP(rud_meth['sali'], press[:, None], cf.lon, cf.lat)
total_sig = gsw.sigma0(total_sa, rud_meth['thetai'])
total_lvls = lvl_profiles(z_a, total_sig, rud_meth['thetai'], cf.sig_lvl)

# sound speed comparison
comp = np.load('data/processed/inputed_decomp.npz')
stable_lvls = comp['stable_lvls']
sig_rud, tau_rud = grid_field(z_a, stable_lvls, cf.sig_lvl[:stable_lvls.shape[1]])
sa_rud, ct_rud = SA_CT_from_sigma0_spiciness0(sig_rud, tau_rud)
"""
c_rud = gsw.sound_speed(sa_rud, ct_rud, press[:, None])

# tau difference method
sec4 = Section()
total_lvls  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_spice = sec4.stable_spice(total_lvls)
stab_lvls = sec4.stable_cntr_height(stab_spice)

sig_bg, tau_bg = grid_field(sec4.z_a, stab_lvls, sec4.sig_lvl)
sig_tilt, tau_tilt = grid_field(sec4.z_a, stab_spice, sec4.sig_lvl)

delta_spice = sec4.spice - tau_tilt
tau_spice = tau_bg + delta_spice

sa_spice, ct_spice = SA_CT_from_sigma0_spiciness0(sig_bg, tau_spice)
c_diff = gsw.sound_speed(sa_spice, ct_spice, press[:, None])

"""

#plot_i = [30, 35, 40]
#plot_i = [46]
plot_i = [48]

fig, ax = plt.subplots()
ax.plot(x_a, smooth_lvls[0, plot_i, :].T, color='0.4')
ax.plot(x_a, total_lvls[0, plot_i, :].T, color='C0')
ax.plot(x_a, stable_lvls[0, plot_i, :].T, color='C2')
ax.set_ylim(150, 0)

# rms c fields
tilt_sa = gsw.SA_from_SP(rud_meth['stilt'], press[:, None], cf.lon, cf.lat)
spice_sa = gsw.SA_from_SP(rud_meth['sspice'], press[:, None], cf.lon, cf.lat)

smooth_c = gsw.sound_speed(smooth_sa, rud_meth['thsmooth'], press[:, None])
tilt_c = gsw.sound_speed(tilt_sa, rud_meth['thtilt'], press[:, None])
spice_c = gsw.sound_speed(spice_sa, rud_meth['thspice'], press[:, None])
total_c = gsw.sound_speed(total_sa, rud_meth['thetai'], press[:, None])

tilt_rms = np.sqrt(np.var(tilt_c - smooth_c, axis=1))
spice_rms = np.sqrt(np.var(spice_c - smooth_c, axis=1))
total_rms = np.sqrt(np.var(total_c - smooth_c, axis=1))

fig, ax = plt.subplots()
ax.plot(tilt_rms, z_a)
ax.plot(spice_rms, z_a)
ax.plot(total_rms, z_a)
ax.set_ylim(150, 0)
ax.set_xlim(0, 2.3)
ax.grid()

fig, ax = plt.subplots()
ax.plot(rms_tilt, z_a)
ax.plot(rms_spice, z_a)
ax.plot(rms_total, z_a)
ax.set_ylim(150, 0)
ax.set_xlim(0, 2.3)
ax.grid()

