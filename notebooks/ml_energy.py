import numpy as np
from math import pi
import matplotlib.pyplot as plt

import os

from src import MLEnergy, MLEnergyPE, list_tl_files

plt.ion()

fc = 400
tl_files = list_tl_files(fc=fc)

def eng_diff(tl_file):
    eng_pe = MLEnergyPE(tl_file)
    eng_mode = MLEnergy(tl_file)
    field_type = 'tilt'

    e_dB_pe = 10 * np.log10(eng_pe.ml_energy(field_type) * eng_pe.r_a)
    e_dB_mode = 10 * np.log10(eng_mode.ml_energy(field_type) * eng_mode.r_a)
    return eng_pe.r_a, e_dB_pe - e_dB_mode

diff = []
fig, ax = plt.subplots()
for tl_file in tl_files:
    r_a, diff_dB = eng_diff(tl_file)
    diff.append(diff_dB)
    ax.plot(r_a / 1e3, diff_dB)
diff = np.array(diff)
ax.set_xlim(5, 45)
ax.set_ylim(-3, 3)

test_i = (r_a > 10e3) & (r_a < 40e3)
np.max(np.abs(diff[:, test_i]), axis=-1)
