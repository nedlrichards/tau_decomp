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

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.0))
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_ri), color='0.6', linestyle='--', label='RI BG')
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_rd), label='RD BG')
ax.plot(ml_eng.r_a / 1e3, 10 * np.log10(eng_mode), label='MLM1')
ax.plot(ml_eng.r_a / 1e3, 20 * np.log10(np.abs(proj_amp)), label='proj. MLM1')
ax.set_xlim(0, 55)
ax.set_ylim(-23, 0)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Energy (dB)')
ax.legend(loc='upper right', bbox_to_anchor=(1.04, 1.06), framealpha=1.0)

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.05
pos.y0 += 0.12
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/m1_project_example.png', dpi=300)

