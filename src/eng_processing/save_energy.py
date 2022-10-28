import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

import pickle

save_dir = 'data/processed/'
range_bounds = (7.5e3, 47.5e3)
cf = Config()

def compute_statistics(fc, source_depth, mode_num=None):
    """statistics used in sparkline plots"""
    eng = EngProc(Config(fc=fc, source_depth=source_depth))
    e_ri = eng.diffraction_bg()
    dyn = eng.dynamic_energy()
    xs = eng.xs

    ml_ml = np.array([e_ri] + [dyn[f] for f in cf.field_types])

    r_a = eng.r_a.copy()
    r_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])

    # Blocking features
    if source_depth == 'shallow':
        d = np.array([dyn[fld] for fld in cf.field_types])
        bg_i = eng.blocking_feature(d, e_ri, range_bounds=range_bounds)
        downrange_block_i = bg_i < 3
        # filter ranges that start below 3 dB down
        demean = d[:, :, r_i] - e_ri[None, :, r_i]
        start_i = demean[:, :, 0] >= -3
        block_i = downrange_block_i & start_i
    else:
        block_i = None

    # project pressure field onto modes
    if mode_num is not None:
        e_ri = eng.diffraction_bg(normalize=False)
        dyn = eng.dynamic_energy(mode_num=mode_num)
        ml_proj = np.array([e_ri] + [np.array(dyn[f]) for f in cf.field_types])
    else:
        ml_proj = None

    # transition layer energy
    if source_depth == 'shallow':
        e_ri = eng.diffraction_bg(int_layer='tl')
        dyn = eng.dynamic_energy(int_layer='tl')
        ml_tl = np.array([e_ri] + [dyn[f] for f in cf.field_types])
    else:
        ml_tl = None

    sf = join(save_dir, 'int_eng_' + source_depth + '_' + str(int(fc)) + '.npz')

    save_dict = {'ml_ml':ml_ml, 'r_a':r_a, 'block_i':block_i,
            'ml_proj':ml_proj, 'ml_tl':ml_tl, 'xs':xs}
    # don't save none values
    np.savez(sf, **{k: v for k, v in save_dict.items() if v is not None})

compute_statistics(400, 'shallow', 1)
print('shallow 400')
compute_statistics(1000, 'shallow', 2)
print('shallow 1000')
compute_statistics(1000, 'deep')
print('deep 1000')
compute_statistics(400, 'deep')
print('deep 400')
