import numpy as np
from math import pi
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from copy import copy
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import find_peaks

import os

from src import RDModes, Config

plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')
bbox = dict(boxstyle='round', fc='w')

fc = 400
z_int = 150.
cf = Config(fc)

run_number = 690
tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                    f'tl_section_{run_number:03}.npz'))
r_a = tl_data['rplot']

def red_eng(field_type):

    rd_modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds, s=None)

    l_len = -2 * pi / np.diff(np.real(rd_modes.k_bg))

    # reference energy
    psi_s = np.exp(1j * pi / 4) / (rd_modes.rho0 * np.sqrt(8 * pi)) \
            * rd_modes.psi_ier(rd_modes.z_src)
    psi_s /= np.sqrt(rd_modes.k_bg)
    psi_s *= 4 * pi

    psi_m0 = psi_s.copy()
    dom_modes = np.zeros(psi_m0.size, dtype=np.bool_)
    am = np.argmax(l_len)

    if l_len[am + 1] > 6e4:
        am = [am, am + 1]
    else:
        am = [am]

    am = np.hstack([[am[0] - 1], am, [am[-1] + 1]])
    dom_modes[am] = True
    psi_m0[~dom_modes] = 0

    p_m0 = rd_modes.synthesize_pressure(psi_m0, z_a, r_synth=r_a)
    en_ri_0 = np.sum(np.abs(p_m0) ** 2, axis=1) * dz

    # reference ml energy
    p_ri = rd_modes.synthesize_pressure(psi_s, z_a, r_synth=r_a)
    en_ri = np.sum(np.abs(p_ri) ** 2, axis=1) * dz

    # reduced mode set estimate of energy
    psi_rd = tl_data[field_type + '_mode_amps'].copy()
    p_rd = rd_modes.synthesize_pressure(psi_rd, z_a, r_synth=r_a)
    en_rd = np.sum(np.abs(p_rd) ** 2, axis=1) * dz

    psi_rd[:, ~dom_modes] = 0
    p_rd = rd_modes.synthesize_pressure(psi_rd, z_a, r_synth=r_a)
    en_rd_0 = np.sum(np.abs(p_rd) ** 2, axis=1) * dz

    return en_ri_0, en_ri, en_rd_0, en_rd

field_type = 'total'

# energy from pe
xs = tl_data['xs']
z_a = tl_data['zplot']
r_a = tl_data['rplot'] - xs
dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
z_i = z_a < z_int
en_pe = np.sum(np.abs(tl_data['p_' + field_type])[:, z_i] ** 2, axis=1) * dz

en_ri_0, _, _, _ = red_eng('bg')
_, _, en_rd_0, en_rd = red_eng(field_type)

ref_dB = 10 * np.log10(en_ri_0)
rd_dB = 10 * np.log10(en_rd_0)
pe_dB = 10 * np.log10(en_pe)

fig, ax = plt.subplots()
ax.plot((r_a + xs) / 1e3, rd_dB - ref_dB)
ax.plot((r_a + xs) / 1e3, pe_dB - ref_dB)

