import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergy, list_tl_files

plt.ion()
source_depth = 'shallow'
#source_depth = 'deep'

def one_freq(fc):
    tl_list = list_tl_files(fc, source_depth=source_depth)

    x_s = []
    e_ri = []
    e_ri_0 = []
    loop_len = []
    for tl in tl_list:
        ml_eng = MLEnergy(tl, source_depth=source_depth, bg_only=True)
        x_s.append(ml_eng.xs)
        e, e_0 = ml_eng.background_diffraction()

        loop_len.append(ml_eng.llen['bg'])
        e_ri.append(e)
        e_ri_0.append(e_0)

    e_ri = np.array(e_ri)
    e_ri_0 = np.array(e_ri_0)
    loop_len = np.array(loop_len, dtype='object')

    r_a = ml_eng.r_a
    x_s = np.array(x_s)
    save_dict = {'r_a':r_a, 'x_s':x_s, 'e_ri':e_ri, 'e_ri_0':e_ri_0,
                 'loop_len':loop_len}

    return save_dict

save_dict = one_freq(400)
save_dict['e_ri_400'] = save_dict.pop('e_ri')
save_dict['e_ri_0_400'] = save_dict.pop('e_ri_0')
save_dict['loop_len_400'] = save_dict.pop('loop_len')
"""
tmp_dict = one_freq(1e3)
save_dict['e_ri_1000'] = tmp_dict.pop('e_ri')
save_dict['e_ri_0_1000'] = tmp_dict.pop('e_ri_0')
save_dict['loop_len_1000'] = tmp_dict.pop('loop_len')
"""
np.savez("data/processed/bg_ri_eng_" + source_depth + ".npz", **save_dict)
