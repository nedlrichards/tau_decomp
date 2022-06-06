import numpy as np
from math import pi
import matplotlib.pyplot as plt

import os

from src import MLEnergy, MLEnergyPE, list_tl_files

plt.ion()

fc = 400

test_i = 50
tl_files = list_tl_files(fc)
tl_file = tl_files[test_i]

eng_pe = MLEnergyPE(tl_file)
eng_mode = MLEnergy(tl_file)
field_type = 'total'


fig, ax = plt.subplots()
ax.plot(eng_pe.r_a / 1e3, 10 * np.log10(eng_pe.ml_energy(field_type) * eng_pe.r_a))
ax.plot(eng_mode.r_a / 1e3, 10 * np.log10(eng_mode.ml_energy(field_type) * eng_mode.r_a))

