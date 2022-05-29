import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergyPE, MLEnergy, Config, list_tl_files
from src import EngProc

plt.ion()
plt.style.use('elr')

fc = 400
#fc = 1e3
source_depth = "shallow"

eng = EngProc(fc=fc, source_depth=source_depth)

cf = eng.cf
r_a = eng.r_a
bg_eng = eng.bg_eng
dynamic_eng = eng.dynamic_eng

range_bounds = (5e3, 50e3)
max_int_loss = eng.blocking_feature(range_bounds=range_bounds,
                                    comp_len=5e3)

m_i = max_int_loss > 3
norm_eng = dynamic_eng - bg_eng

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

ax[0].text(5, 3, eng.dy_fields[0], bbox=cf.bbox)
ax[1].text(5, 3, eng.dy_fields[1], bbox=cf.bbox)
ax[2].text(5, 3, eng.dy_fields[2], bbox=cf.bbox)

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

p_total_stats = eng.field_stats(norm_eng[0, ~m_i[0], :])
p_tilt_stats = eng.field_stats(norm_eng[1, ~m_i[1], :])
p_spice_stats = eng.field_stats(norm_eng[2, ~m_i[2], :])

total_stats = eng.field_stats(norm_eng[0])
tilt_stats = eng.field_stats(norm_eng[1])
spice_stats = eng.field_stats(norm_eng[2])

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

ax[0].legend(eng.dy_fields)
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

