import numpy as np
from math import pi
import matplotlib.pyplot as plt

import os

from src import Config, list_tl_files, RDModes

plt.ion()

fc = 400
tl_files = list_tl_files(fc)
field_type = 'spice'
#field_type = 'total'
#field_type = 'tilt'

cf = Config(fc)

def modeamp(runfile):
    tl_data = np.load(runfile)

    modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds, s=None)

    x_a = tl_data['r_modes']
    eps = np.spacing(1)
    llen = -2 * pi / (np.diff(np.real(modes.k_bg)) - eps)

    am = np.argmax(llen)
    mode_nums = [am - 1, am, am + 1]
    int_i = modes.z_a < cf.z_int
    ml_mode = modes.psi_bg[:, int_i][mode_nums, :]
    max_i = np.argmax(np.sum(ml_mode, axis=1))
    am = mode_nums[max_i]

    amps_m1 = tl_data[field_type + '_mode_amps'][:, am]
    amps_dB = 20 * np.log10(np.abs(amps_m1))
    amps_dB -= amps_dB[0]
    if np.max(amps_dB) > 5:
        amps_dB = np.full_like(amps_dB, np.nan)
    return amps_dB

amps_dB = []
for tl_file in tl_files:
    amps_dB.append(modeamp(tl_file))
amps_dB = np.array(amps_dB)

x_a = np.load(tl_files[0])['r_modes']

fig, ax = plt.subplots()
ax.grid()
ax.plot(x_a, amps_dB.T, color='0.6', alpha=0.6)
ax.plot(x_a, np.nanmean(amps_dB, axis=0), color='0.2')

ax.set_ylim(-20, 5)
ax.set_xlim(0, 60)
ax.set_ylabel('Mode energy')
ax.set_xlabel('Range (km)')

pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y1 += 0.04
pos.y0 += 0.04
ax.set_position(pos)
