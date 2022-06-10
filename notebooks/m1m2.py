import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

from src import MLEnergy, MLEnergyPE, list_tl_files, Config

plt.ion()

fc = 400
tl_files = list_tl_files(fc=fc)

file_i = 10
field_type = 'total'
percent_max = .1

def ri_diff(tl_file):
    eng_mode = MLEnergy(tl_file, m1_percent=10)
    cnt = eng_mode.set_1['total'].size
    en_ri, en_ri_0 = eng_mode.background_diffraction(field_type)
    en_rd = eng_mode.ml_energy(field_type, indicies=None)
    en_rd_0 = eng_mode.ml_energy(field_type, indicies=eng_mode.set_1[field_type])
    en_ri_dB = 10 * np.log10(en_ri * eng_mode.r_a)
    en_ri_0_dB = 10 * np.log10(en_ri_0 * eng_mode.r_a)
    en_rd_dB = 10 * np.log10(en_rd * eng_mode.r_a)
    en_rd_0_dB = 10 * np.log10(en_rd_0 * eng_mode.r_a)

    return eng_mode.r_a, en_ri_0_dB - en_ri_dB, en_rd_0_dB - en_rd_dB

diff_ri = []
diff_rd = []
m1_count = []
for tl_file in tl_files:
    r_a, diff_eng, diff_eng_rd = ri_diff(tl_file)
    diff_ri.append(diff_eng)
    diff_rd.append(diff_eng_rd)

diff_ri = np.array(diff_ri)
diff_rd = np.array(diff_rd)

fig, ax = plt.subplots()
ax.plot(r_a / 1e3, diff_rd[1].T)
"""
x_t_i = np.argmin(np.abs(r_a - 37e3))
diff = np.array(diff)

fig, ax = plt.subplots()
tl_file = tl_files[3]
r_a, diff_eng, cnt = ri_diff(tl_file)

fig, ax = plt.subplots()
ax.plot(r_a / 1e3, diff_eng)

eng_mode = MLEnergy(tl_file)
ms1 = eng_mode.mode_set_1(field_type)

fig, ax = plt.subplots()
ax.plot(eng_mode.field_modes[field_type].psi_bg[ms1, :].T, eng_mode.z_a_modes)
ax.set_ylim(150, 0)
"""
