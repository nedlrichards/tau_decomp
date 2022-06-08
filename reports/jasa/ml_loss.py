import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src import MLEnergyPE, MLEnergy, Config, list_tl_files
from src import EngProc

plt.ion()
plt.style.use('elr')

fc = 400
#fc = 1e3
source_depth = "shallow"

cf = Config(fc=fc, source_depth=source_depth)
tl_files = list_tl_files(fc=fc)

def ml_energy(tl_file):
    eng_mode = MLEnergy(tl_file)
    e_diffr = 10 * np.log10(eng_mode.background_diffraction() * eng_mode.r_a)
    dyn_eng = []
    for ft in cf.field_types:
        dyn_eng.append(10 * np.log10(eng_mode.ml_energy(ft) * eng_mode.r_a))
    return eng_mode.r_a, e_diffr, dyn_eng

bg_eng = []
dynamic_eng = []

for tl in tl_files:
    r_a, bg, dy = ml_energy(tl)
    bg_eng.append(bg)
    dynamic_eng.append(dy)

bg_eng = np.array(bg_eng)
bg_eng = bg_eng[:, 0, :]
dynamic_eng = np.array(dynamic_eng)

range_bounds = (5e3, 50e3)

norm_eng = dynamic_eng - bg_eng[:, None, :]

# threshold loss features
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(cf.jasa_1clm,3))
ax[0].plot(r_a / 1e3, norm_eng[0, ~m_i[0], :].T, linewidth=.5, color='0.4')
ax[0].plot(r_a / 1e3, norm_eng[0, m_i[0], :].T, linewidth=.5, color='C1')

ax[1].plot(r_a / 1e3, norm_eng[1, ~m_i[1], :].T, linewidth=.5, color='0.4')
ax[1].plot(r_a / 1e3, norm_eng[1, m_i[1], :].T, linewidth=.5, color='C1')

ax[2].plot(r_a / 1e3, norm_eng[2, ~m_i[2], :].T, linewidth=.5, color='0.4')
ax[2].plot(r_a / 1e3, norm_eng[2, m_i[2], :].T, linewidth=.5, color='C1')

ax[0].set_xlim(5, 45)
ax[1].set_ylim(-15, 5)

ax[0].text(5, 3, cf.field_types[1], bbox=cf.bbox)
ax[1].text(5, 3, cf.field_types[2], bbox=cf.bbox)
ax[2].text(5, 3, cf.field_types[3], bbox=cf.bbox)

fig.supylabel('ML energy (db re Background)')
ax[2].set_xlabel('Range (km)')

pos = ax[0].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[2].set_position(pos)

#fig.savefig('reports/jasa/figures/ml_energy.png', dpi=300)

def field_stats(r_a, field_eng, range_bounds=(5e3, 50e3)):
    """common statistics taken over field realization"""
    r_i = (r_a > range_bounds[0]) & (r_a < range_bounds[1])

    f_mean = np.mean(field_eng[:, r_i], axis=0)
    f_rms = np.sqrt(np.var(field_eng[:, r_i], axis=0))
    f_10 = np.percentile(field_eng[:, r_i], 10, axis=0,
                            method='median_unbiased')
    f_90 = np.percentile(field_eng[:, r_i], 90, axis=0,
                            method='median_unbiased')

    r_a = r_a[r_i]
    f_mean_rgs = linregress(r_a, y=f_mean)
    f_rms_rgs = linregress(r_a, y=f_mean + f_rms)
    f_10_rgs = linregress(r_a, y=f_10)
    f_90_rgs = linregress(r_a, y=f_90)

    stats = {"r_a":r_a, 'mean':f_mean, 'rms':f_rms,
                '10th':f_10, '90th':f_90,
                'mean_rgs':f_mean_rgs, 'rms_rgs':f_rms_rgs,
                '10th_rgs':f_10_rgs, '90th_rgs':f_90_rgs}
    return stats




p_total_stats = field_stats(r_a, norm_eng[0, ~m_i[0], :])
p_tilt_stats = field_stats(r_a, norm_eng[1, ~m_i[1], :])
p_spice_stats = field_stats(r_a, norm_eng[2, ~m_i[2], :])

total_stats = field_stats(r_a, norm_eng[0])
tilt_stats = field_stats(r_a, norm_eng[1])
spice_stats = field_stats(r_a, norm_eng[2])

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(cf.jasa_2clm,3))

ax[0].plot(p_total_stats['r_a'] / 1e3, p_total_stats['mean'], linewidth=3)
ax[0].plot(p_tilt_stats['r_a'] / 1e3, p_tilt_stats['mean'], linewidth=3)
ax[0].plot(p_spice_stats['r_a'] / 1e3, p_spice_stats['mean'], linewidth=3)

ax[0].plot(p_total_stats['r_a'] / 1e3, p_total_stats['rms'],
           linewidth=1, color='C0')
ax[0].plot(p_tilt_stats['r_a'] / 1e3, p_tilt_stats['rms'],
           linewidth=1, color='C1')
ax[0].plot(p_spice_stats['r_a'] / 1e3, p_spice_stats['rms'],
           linewidth=1, color='C2')
ax[0].plot(p_total_stats['r_a'] / 1e3,
           2 * p_total_stats['mean'] - p_total_stats['rms'],
           linewidth=1, color='C0')
ax[0].plot(p_tilt_stats['r_a'] / 1e3,
           2 * p_tilt_stats['mean'] - p_tilt_stats['rms'],
           linewidth=1, color='C1')
ax[0].plot(p_spice_stats['r_a'] / 1e3,
           2 * p_spice_stats['mean'] - p_spice_stats['rms'],
           linewidth=1, color='C2')

ax[0].legend(cf.field_types[1:])
"""
ax[0].plot(p_total_stats['r_a'] / 1e3, p_total_stats['10th'],
           linewidth=1, linestyle='--', color='C0')
ax[0].plot(p_tilt_stats['r_a'] / 1e3, p_tilt_stats['10th'],
           linewidth=1, linestyle='--', color='C1')
ax[0].plot(p_spice_stats['r_a'] / 1e3, p_spice_stats['10th'],
           linewidth=1, linestyle='--', color='C2')
ax[0].plot(p_total_stats['r_a'] / 1e3, p_total_stats['90th'],
           linewidth=1, linestyle='--', color='C0')
ax[0].plot(p_tilt_stats['r_a'] / 1e3, p_tilt_stats['90th'],
           linewidth=1, linestyle='--', color='C1')
ax[0].plot(p_spice_stats['r_a'] / 1e3, p_spice_stats['90th'],
           linewidth=1, linestyle='--', color='C2')
"""

ax[1].plot(total_stats['r_a'] / 1e3, total_stats['mean'], linewidth=3)
ax[1].plot(tilt_stats['r_a'] / 1e3, tilt_stats['mean'], linewidth=3)
ax[1].plot(spice_stats['r_a'] / 1e3, spice_stats['mean'], linewidth=3)

ax[1].plot(p_total_stats['r_a'] / 1e3, total_stats['rms'],
           linewidth=1, color='C0')
ax[1].plot(p_tilt_stats['r_a'] / 1e3, tilt_stats['rms'],
           linewidth=1, color='C1')
ax[1].plot(p_spice_stats['r_a'] / 1e3, spice_stats['rms'],
           linewidth=1, color='C2')
ax[1].plot(p_total_stats['r_a'] / 1e3,
           2 * total_stats['mean'] - total_stats['rms'],
           linewidth=1, color='C0')
ax[1].plot(p_tilt_stats['r_a'] / 1e3,
           2 * tilt_stats['mean'] - tilt_stats['rms'],
           linewidth=1, color='C1')
ax[1].plot(p_spice_stats['r_a'] / 1e3,
           2 * spice_stats['mean'] - spice_stats['rms'],
           linewidth=1, color='C2')

ax[1].plot(p_total_stats['r_a'] / 1e3, total_stats['10th'],
           linewidth=1, linestyle='--', color='C0')
ax[1].plot(p_tilt_stats['r_a'] / 1e3, tilt_stats['10th'],
           linewidth=1, linestyle='--', color='C1')
ax[1].plot(p_spice_stats['r_a'] / 1e3, spice_stats['10th'],
           linewidth=1, linestyle='--', color='C2')

ax[1].plot(p_total_stats['r_a'] / 1e3, total_stats['90th'],
           linewidth=1, linestyle='--', color='C0')
ax[1].plot(p_tilt_stats['r_a'] / 1e3, tilt_stats['90th'],
           linewidth=1, linestyle='--', color='C1')
ax[1].plot(p_spice_stats['r_a'] / 1e3, spice_stats['90th'],
           linewidth=1, linestyle='--', color='C2')


ax[0].text(7, 5, 'W/O blocking', bbox=cf.bbox)
ax[1].text(7, 5, 'With blocking', bbox=cf.bbox)

ax[0].set_xlim(5, 45)
ax[0].set_ylim(-15, 5)

ax[0].set_ylabel('ML energy (dB re Background)')
fig.supxlabel('Position, $x$ (km)')

pos = ax[0].get_position()
pos.x0 += 0.00
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += 0.02
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[1].set_position(pos)

#fig.savefig('reports/jasa/figures/ml_energy_stats.png', dpi=300)

