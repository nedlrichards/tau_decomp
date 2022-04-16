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

def ml_energy(run_number):
    tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                        f'tl_section_{run_number:03}.npz'))
    r_a = tl_data['rplot']
    rd_modes = RDModes(tl_data['c_bg'], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds, s=None)

    xs = tl_data['xs']
    dr = (rd_modes.r_plot[-1] - rd_modes.r_plot[0]) / (rd_modes.r_plot.size - 1)
    r_max = 60e3
    num_r = int(np.ceil(r_max / dr))
    r_a_modes = (np.arange(num_r) + 1) * dr

    l_len = -2 * pi / np.diff(np.real(rd_modes.k_bg))

    # reference energy
    psi_s = np.exp(1j * pi / 4) / (rd_modes.rho0 * np.sqrt(8 * pi)) \
            * rd_modes.psi_ier(rd_modes.z_src)
    psi_s /= np.sqrt(rd_modes.k_bg)
    psi_s *= 4 * pi

    z_a = tl_data['zplot']
    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

    p_ri = rd_modes.synthesize_pressure(psi_s, z_a, r_synth=r_a_modes)
    en_ri = np.sum(np.abs(p_ri) ** 2, axis=1) * dz

    psi_m0 = psi_s.copy()
    # Mode 1s seem to get the general trend
    #dom_modes = rd_modes.mode_number == 0
    # Mode 1 and 2s nail the trend
    dom_modes = (rd_modes.mode_number == 0) | (rd_modes.mode_number == 1)

    # either 3 or 4 selected modes
    dom_modes = np.zeros_like(dom_modes)
    am = np.argmax(l_len)

    if l_len[am + 1] > 6e4:
        am = [am, am + 1]
    else:
        am = [am]

    am = np.hstack([[am[0] - 1], am, [am[-1] + 1]])
    dom_modes[am] = True
    psi_m0[~dom_modes] = 0

    p_m0 = rd_modes.synthesize_pressure(psi_m0, z_a, r_synth=r_a_modes)
    en_m0 = np.sum(np.abs(p_m0) ** 2, axis=1) * dz
    return r_a_modes, en_m0

diff_eng = []
for rn in np.arange(70) * 10:
    r_a_modes, ml_eng = ml_energy(rn)
    diff_eng.append(ml_eng)
diff_eng = np.array(diff_eng)

eng_dB = 10 * np.log10(diff_eng * r_a_modes).T
fig, ax = plt.subplots()
ax.plot(r_a_modes / 1e3, eng_dB, '0.4', linewidth=0.5)
ax.grid()

