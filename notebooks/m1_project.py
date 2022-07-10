import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

from src import MLEnergy, list_tl_files, Config
from src import RDModes, section_cfield

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)
field_type = 'bg'

tl_file = tl_files[14]
ml_eng = MLEnergy(tl_file)

eng_rd = ml_eng.ml_energy(field_type) * ml_eng.r_a
eng_ri, _ = ml_eng.background_diffraction(field_type) * ml_eng.r_a

ind = ml_eng.mode_set_1(field_type, m1_percent=99.9)
eng_mode = np.sum(ml_eng.tl_data[field_type + '_mode_amps'][:, ind] ** 2, axis=-1) * 1e3
proj_amp = ml_eng.proj_mode1(field_type)

fig, ax = plt.subplots()
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_ri), color='0.6', linestyle='--')
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_rd))
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_mode))
ax.plot(ml_eng.r_a / 1e3, 20 * np.log10(np.abs(proj_amp)))
ax.set_xlim(0, 55)
ax.set_ylim(-23, -5)

def diff_eng(tl_file):
    """difference between methods"""
    r_bound = (10e3, 40e3)
    ml_eng = MLEnergy(tl_file)
    r_i = (ml_eng.r_a > r_bound[0]) & (ml_eng.r_a < r_bound[-1])
    proj_eng = np.abs(ml_eng.proj_mode1(field_type)) ** 2
    pe_eng = ml_eng.ml_energy(field_type) * ml_eng.r_a
    diff_eng = 10 * (np.log10(proj_eng) - np.log10(pe_eng))

    return np.mean(diff_eng[r_i])

"""
de = np.array([diff_eng(tl) for tl in tl_files])

de_m = np.mean(de)
de_rms = np.var(de)
de_10 = np.percentile(de, 25, method='median_unbiased')
de_90 = np.percentile(de, 75, method='median_unbiased')

fig, ax = plt.subplots()
ax.plot(de, '.')
ax.plot(np.full_like(de, de_m), color='k', linewidth=3)
ax.plot(np.full_like(de, de_m + de_rms), '--', color='k', linewidth=1)
ax.plot(np.full_like(de, de_m - de_rms), '--', color='k', linewidth=1)
ax.plot(np.full_like(de, de_10), color='C1', linewidth=1)
ax.plot(np.full_like(de, de_90), color='C1', linewidth=1)
"""

