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
mode_num = 2

tl_file = tl_files[11]

eng_mode = MLEnergy(tl_file)

total_eng_dB = 10 * np.log10(eng_mode.ml_energy('bg'))
eng_bg = eng_mode.background_diffraction('bg')

ref_eng = 10 * np.log10(eng_bg)
ref_eng_comp = 10 * np.log10(eng_bg * eng_mode.r_a)

proj_amp, proj_scale = eng_mode.proj_mode('bg', mode_num=2)
proj_eng = np.abs(proj_amp) ** 2 / proj_scale

fig, ax = plt.subplots()
ax.plot(eng_mode.r_a / 1e3, total_eng_dB - ref_eng)
ax.plot(eng_mode.r_a / 1e3, 10 * np.log10(proj_eng) - ref_eng_comp)

