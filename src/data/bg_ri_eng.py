import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergy, Config, list_tl_files

plt.ion()

fc = 400
cf = Config(fc)
tl_list = list_tl_files(fc)

x_s = []
e_ri = []
e_ri_0 = []
loop_len = []
for tl in tl_list:
    ml_eng = MLEnergy(fc, tl)
    x_s.append(ml_eng.xs)
    ll = -2 * pi / np.diff(np.real(ml_eng.field_modes['bg'].k_bg))
    e, e_0 = ml_eng.background_diffraction()

    loop_len.append(ll)
    e_ri.append(e)
    e_ri_0.append(e_0)

r_a = ml_eng.r_a
x_s = np.array(x_s)
e_ri = np.array(e_ri)
e_ri_0 = np.array(e_ri_0)
loop_len = np.array(loop_len, dtype='object')

save_dict = {'r_a':r_a, 'x_s':x_s, 'e_ri':e_ri, 'e_ri_0':e_ri_0,
             'loop_len':loop_len}
np.savez('data/processed/bg_ri_eng.npz', **save_dict)
