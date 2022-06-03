import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergyPE, MLEnergy, Config, list_tl_files
from src import EngProc

plt.ion()
plt.style.use('elr')

fc = 400
source_depth = "shallow"

cf = Config(fc=fc, source_depth=source_depth)
eng = EngProc(cf)

range_bounds = (5e3, 50e3)
max_int_loss = eng.blocking_feature(range_bounds=range_bounds,
                                    comp_len=5e3)

m_i = max_int_loss > 3

r_a = eng.r_a
bg_eng = eng.bg_eng
dynamic_eng = eng.dynamic_eng


save_dict = {"m_i":m_i, "xs":eng.xs, "bg_eng_400":eng.bg_eng,
             "dynamic_eng_400":eng.dynamic_eng, "r_a":eng.r_a,
             "max_int_loss_400":max_int_loss}

fc = 1e3
source_depth = "shallow"

cf = Config(fc=fc, source_depth=source_depth)
eng = EngProc(cf)
max_int_loss = eng.blocking_feature(range_bounds=range_bounds,
                                    comp_len=5e3)
save_dict["bg_eng_1000"] = eng.bg_eng
save_dict["dynamic_eng_1000"] = eng.dynamic_eng
save_dict["max_int_loss_1000"] = max_int_loss

np.savez('data/processed/energy_processing.npz', **save_dict)
