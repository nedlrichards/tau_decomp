import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

from src import MLEnergy, list_tl_files, Config
from src import RDModes, section_cfield

plt.ion()
plt.style.use('elr')

fc = 1000
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)

tl_file = tl_files[23]
ft = 'total'

eng_mode = MLEnergy(tl_file)
rd_modes = eng_mode.field_modes[ft]

p_pe = eng_mode.tl_data['p_' + ft][:, eng_mode.z_i]
p_modes = rd_modes.synthesize_pressure(eng_mode.tl_data[ft + '_mode_amps'],
                                       eng_mode.z_a[eng_mode.z_i],
                                       r_synth=eng_mode.r_a)

# recreate ML with limited modes
mode_num = 1
mn = []
for i in [1, 2, 3]:
    mn.append(eng_mode.mode_set(ft, m1_percent=99., mode_num=i)[0])

sel_amps = np.zeros_like(eng_mode.tl_data[ft + '_mode_amps'])
sel_amps[:, mn] = eng_mode.tl_data[ft + '_mode_amps'][:, mn]

p_sel_modes = rd_modes.synthesize_pressure(sel_amps,
                                           eng_mode.z_a[eng_mode.z_i],
                                           r_synth=eng_mode.r_a)


cmap = cf.cmap
zplot = eng_mode.z_a[eng_mode.z_i]
rplot = eng_mode.r_a
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
cm = axes[0].pcolormesh(rplot / 1e3, zplot, 20 * np.log10(np.abs(p_pe)).T,
                        cmap=cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[1].pcolormesh(rplot / 1e3, zplot, 20 * np.log10(np.abs(p_modes)).T,
                        cmap=cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[2].pcolormesh(rplot / 1e3, zplot, 20 * np.log10(np.abs(p_sel_modes)).T,
                        cmap=cmap, vmax=-50, vmin=-90, rasterized=True)

dz = (zplot[-1] - zplot[0]) / (zplot.size - 1)

fig, ax = plt.subplots()
int_eng = np.sum(np.abs(p_pe) ** 2, axis=-1) * dz
ax.plot(rplot / 1e3, 10 * np.log10(int_eng * rplot))
int_eng = np.sum(np.abs(p_modes) ** 2, axis=-1) * dz
ax.plot(rplot / 1e3, 10 * np.log10(int_eng * rplot))
int_eng = np.sum(np.abs(p_sel_modes) ** 2, axis=-1) * dz
ax.plot(rplot / 1e3, 10 * np.log10(int_eng * rplot))

# energy content by mode number
mode_en = []
for i in [1, 2, 3]:
    mn = eng_mode.mode_set(ft, m1_percent=99., mode_num=i)[0]

    sel_amps = np.zeros_like(eng_mode.tl_data[ft + '_mode_amps'])
    sel_amps[:, mn] = eng_mode.tl_data[ft + '_mode_amps'][:, mn]

    p_sel = rd_modes.synthesize_pressure(sel_amps,
                                            eng_mode.z_a[eng_mode.z_i],
                                            r_synth=eng_mode.r_a)

    mode_en.append(np.sum(np.abs(p_sel) ** 2, axis=-1) * dz)

en_dB = 10 * np.log10(mode_en)
en_ref = 10 * np.log10(int_eng)

#int_eng = np.sum(np.abs(p_pe) ** 2, axis=-1) * dz
int_eng = np.sum(np.abs(p_sel_modes) ** 2, axis=-1) * dz

fig, ax = plt.subplots()
#ax.plot(rplot / 1e3, (mode_en / int_eng).T)
ax.plot(rplot / 1e3, (mode_en / int_eng).T)


