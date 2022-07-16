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

tl_files = tl_files[21:24]
ft = 'total'

def relative_eng(tl_file):
    eng_mode = MLEnergy(tl_file)
    rd_modes = eng_mode.field_modes[ft]

    # recreate ML with limited modes
    mn = []
    for i in [1, 2, 3]:
        mn.append(eng_mode.mode_set(ft, m1_percent=99., mode_num=i)[0])

    sel_amps = np.zeros_like(eng_mode.tl_data[ft + '_mode_amps'])
    sel_amps[:, mn] = eng_mode.tl_data[ft + '_mode_amps'][:, mn]

    p_sel_modes = rd_modes.synthesize_pressure(sel_amps,
                                            eng_mode.z_a[eng_mode.z_i],
                                            r_synth=eng_mode.r_a)
    z_a = eng_mode.z_a[eng_mode.z_i]
    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

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

    eng_sel = np.sum(np.abs(p_sel_modes) ** 2, axis=-1) * dz

    r_plot = eng_mode.r_a

    return r_plot, eng_sel, mode_en

rel_eng = []
for tf in tl_files:
    r_a, eng_sel, mode_en = relative_eng(tf)
    rel_eng.append(mode_en / eng_sel)

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 6))
axes[0].plot(r_a / 1e3, rel_eng[0].T)
axes[0].plot([33, 33], [0.7, 1], color='r')
axes[0].plot([47, 47], [0.7, 1], color='r')
axes[0].text(0, 0.8, tl_files[0].split('_')[-1].split('.')[0] + 'km')

axes[1].plot(r_a / 1e3, rel_eng[1].T)
axes[1].plot([23, 23], [0.7, 1], color='r')
axes[1].plot([37, 37], [0.7, 1], color='r')
axes[1].text(0, 0.8, tl_files[1].split('_')[-1].split('.')[0] + 'km')

axes[2].plot(r_a / 1e3, rel_eng[2].T)
axes[2].plot([13, 13], [0.7, 1], color='r')
axes[2].plot([27, 27], [0.7, 1], color='r')
axes[2].text(0, 0.8, tl_files[2].split('_')[-1].split('.')[0] + ' km')

axes[0].legend(labels=['M 1', 'M 2', 'M 3'], ncol=3, loc=(0.2, 1.1))

axes[2].set_xlabel('Source range (km)')
fig.supylabel('Mode energy %')
