"""helper functions"""

import numpy as np
from os import listdir
from os.path import join

def list_tl_files(fc):
    """List all files at a partiular frequency"""
    tl_dir = f'data/processed/field_{int(fc)}'
    run_list = listdir(tl_dir)
    tl_list = [r for r in run_list if r.split('_')[0] == 'tl']
    tl_list.sort()
    full_path = [join(tl_dir, f) for f in tl_list]
    return full_path


def sonic_layer_depth(z_c, c_field, z_max=None, x_intrp=None):
    """
    Return depth of ML maximum
    x_intrp is [x_c, x_new]
    """
    if z_max is not None:
        z_i = z_c <= z_max
    else:
        z_i = np.ones(z_c.size, dtype=np.bool_)

    i_sld = np.argmax(c_field[z_i, :], axis=0)
    z_sld = np.take(z_c[z_i], i_sld)

    if x_intrp is not None:
        z_sld = np.interp(x_intrp[1], x_intrp[0], z_sld)
    return z_sld, i_sld
