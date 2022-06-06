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

proc_eng = np.load('data/processed/energy_processing.npz')

r_a = proc_eng['r_a']
bg_eng_400 = proc_eng["bg_eng_400"]
dynamic_eng_400 = proc_eng["dynamic_eng_400"]


fig, ax = plt.subplots()
ax.plot(eng_pe.r_a / 1e3, 10 * np.log10(eng_pe.ml_energy('bg') * eng_pe.r_a))
ax.plot(r_a / 1e3, bg_eng_400[test_i, :])

fig, ax = plt.subplots()
ax.plot(eng_pe.r_a / 1e3, 10 * np.log10(eng_pe.ml_energy('total') * eng_pe.r_a))
ax.plot(r_a / 1e3, dynamic_eng_400[-1, test_i, :])

