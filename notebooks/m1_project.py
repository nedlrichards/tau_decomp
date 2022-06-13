import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

from src import MLEnergy, MLEnergyPE, list_tl_files, Config
from src import RDModes, section_cfield

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)

percent_max = 99.99
field_type = 'bg'

tl_file = tl_files[60]

tl_data = np.load(tl_file)
p_field = tl_data['p_' + field_type]
z_a = tl_data['zplot']
r_a = tl_data['rplot'] - tl_data['xs']
dz_pe = (z_a[-1] - z_a[0]) / (z_a.size - 1)

fields = np.load('data/processed/inputed_decomp.npz')

c_field = fields['c_' + field_type]
x_a, c_sec = section_cfield(tl_data['xs'], fields['x_a'], c_field)
psi_k = (tl_data['psi_' + field_type], tl_data['k_' + field_type])
rd_modes = RDModes(c_sec, x_a, fields['z_a'], cf, psi_k_bg=psi_k)

eng_field = np.sum(np.abs(p_field) ** 2, axis=-1) * dz_pe

r_test = np.arange(60) * 1e3
dz = (tl_data['z_a'][-1] - tl_data['z_a'][0]) / (tl_data['z_a'].size - 1)
p_test = rd_modes.synthesize_pressure(tl_data[field_type + "_mode_amps"],
                                      tl_data['z_a'], r_synth=r_test)
eng_test = np.sum(np.abs(p_test) ** 2, axis=-1) * dz

eng_mode = np.sum(np.abs(tl_data[field_type + '_mode_amps']) ** 2, axis=-1) * 1e3

fig, ax = plt.subplots()
#ax.plot(r_a / 1e3, 10 * np.log10(eng_field * r_a))
ax.plot(r_test / 1e3, 10 * np.log10(eng_test * r_test))
ax.plot(r_a / 1e3, 10 * np.log10(eng_mode))

eng_mode = MLEnergy(tl_file, m1_percent=percent_max)
ind = eng_mode.set_1[field_type]
psi_1 = eng_mode.field_modes[field_type].psi_bg[ind]

psi_ier = interp1d(tl_data['z_a'], np.squeeze(psi_1))
psi_proj = psi_ier(z_a)
z0_ind = np.argmax(np.abs(np.diff(np.sign(psi_proj))) > 1.5)

eng_field = np.sum(np.abs(p_field[:, :z0_ind]) ** 2, axis=-1) * dz_pe
p_test = rd_modes.synthesize_pressure(tl_data[field_type + "_mode_amps"], z_a[:z0_ind],
                                      r_synth=r_a)
eng_test = np.sum(np.abs(p_test) ** 2, axis=-1) * dz_pe
eng_mode = np.sum(tl_data[field_type + '_mode_amps'][:, ind] ** 2, axis=-1) * 1e3

proj_amp = np.sum(p_field[:, :z0_ind] * psi_proj[None, :z0_ind], axis=-1) * np.sqrt(r_a) * dz_pe / 1e3
eng_proj = np.abs(proj_amp) ** 2 * 1e3

fig, ax = plt.subplots()
ax.plot(r_a / 1e3, 10 * np.abs(eng_field * r_a))
ax.plot(r_a / 1e3, 10 * np.abs(eng_test * r_a))
ax.plot(r_a / 1e3, 10 * np.abs(eng_mode))
ax.plot(r_a / 1e3, 10 * np.abs(eng_proj))
