import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os

from src import MLEnergy, list_tl_files, Config
from src import RDModes, section_cfield

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)
field_type = 'bg'

r_bound = (7.5e3, 47.5e3)

tl_file = tl_files[15]
ml_eng = MLEnergy(tl_file)

r_i = (ml_eng.r_a > r_bound[0]) & (ml_eng.r_a < r_bound[-1])

ind = ml_eng.mode_set_1(field_type, m1_percent=99.9)

mode_eng = np.abs(ml_eng.tl_data[field_type + '_mode_amps'][:, ind[0]]) ** 2 * 1e3
mode_eng = 10 * np.log10(mode_eng)

proj_eng = np.abs(ml_eng.proj_mode1(field_type)) ** 2
proj_eng = 10 * np.log10(proj_eng)

pe_eng = ml_eng.ml_energy(field_type) * ml_eng.r_a
pe_eng = 10 * np.log10(pe_eng)

# linregression
mode_lrg = linregress(ml_eng.r_a[r_i], y=mode_eng[r_i])
proj_lrg = linregress(ml_eng.r_a[r_i], y=proj_eng[r_i])
pe_lrg = linregress(ml_eng.r_a[r_i], y=pe_eng[r_i])

mode_fit = mode_lrg.intercept + mode_lrg.slope * ml_eng.r_a
proj_fit = proj_lrg.intercept + proj_lrg.slope * ml_eng.r_a
pe_fit = pe_lrg.intercept + pe_lrg.slope * ml_eng.r_a

"""
fig, ax = plt.subplots()
ax.plot(ml_eng.r_a / 1e3, pe_eng)
ax.plot(ml_eng.r_a / 1e3, mode_eng)
ax.plot(ml_eng.r_a / 1e3, proj_eng)

ax.plot(ml_eng.r_a / 1e3, mode_fit)
ax.plot(ml_eng.r_a / 1e3, proj_fit)
ax.plot(ml_eng.r_a / 1e3, pe_fit)

fig, ax = plt.subplots()
ax.plot(ml_eng.r_a[r_i] / 1e3,  (mode_fit - mode_eng)[r_i])
ax.plot(ml_eng.r_a[r_i] / 1e3,  (proj_fit - proj_eng)[r_i])
ax.plot(ml_eng.r_a[r_i] / 1e3,  (pe_fit - pe_eng)[r_i])

"""

def diff_eng(tl_file):
    """difference between methods"""
    ml_eng = MLEnergy(tl_file)
    r_i = (ml_eng.r_a > r_bound[0]) & (ml_eng.r_a < r_bound[-1])
    ind = ml_eng.mode_set_1(field_type, m1_percent=99.9)

    mode_eng = np.abs(ml_eng.tl_data[field_type + '_mode_amps'][:, ind[0]])
    mode_eng = 10 * np.log10(mode_eng ** 2 * 1e3)
    proj_eng = 10 * np.log10(np.abs(ml_eng.proj_mode1(field_type)) ** 2)
    pe_eng = 10 * np.log10(ml_eng.ml_energy(field_type) * ml_eng.r_a)

    # linregression
    mode_lrg = linregress(ml_eng.r_a[r_i], y=mode_eng[r_i])
    proj_lrg = linregress(ml_eng.r_a[r_i], y=proj_eng[r_i])
    pe_lrg = linregress(ml_eng.r_a[r_i], y=pe_eng[r_i])

    int_diff = np.array([proj_lrg.intercept - pe_lrg.intercept,
                        mode_lrg.intercept - proj_lrg.intercept])

    slope_diff = np.array([proj_lrg.slope - pe_lrg.slope,
                           mode_lrg.slope - proj_lrg.slope])

    mode_fit = mode_lrg.intercept + mode_lrg.slope * ml_eng.r_a[r_i]
    proj_fit = proj_lrg.intercept + proj_lrg.slope * ml_eng.r_a[r_i]
    pe_fit = pe_lrg.intercept + pe_lrg.slope * ml_eng.r_a[r_i]

    fit_diff = np.array([np.var(mode_fit - mode_eng[r_i]),
                         np.var(proj_fit - proj_eng[r_i]),
                         np.var(pe_fit - pe_eng[r_i])])


    return int_diff, slope_diff, fit_diff

int_diff = []
slope_diff = []
fit_diff = []
for tl in tl_files:
    ir, sr, fr = diff_eng(tl)
    int_diff.append(ir)
    slope_diff.append(sr)
    fit_diff.append(fr)

int_diff = np.array(int_diff)
slope_diff = np.array(slope_diff)
fit_diff = np.array(fit_diff)

fig, ax = plt.subplots()
ax.plot(int_diff[:, 0], '.')
ax.plot(int_diff[:, 1], '.')

fig, ax = plt.subplots()
ax.plot(slope_diff[:, 0], '.')
ax.plot(slope_diff[:, 1], '.')

fig, ax = plt.subplots()
ax.plot(fit_diff[:, 0], '.')
ax.plot(fit_diff[:, 1], '.')
ax.plot(fit_diff[:, 2], '.')
