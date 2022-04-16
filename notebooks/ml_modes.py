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

#x_start = 130
x_start = 40

#def plot(x_start):
run_number = int(x_start)
tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                f'tl_section_{run_number:03}.npz'))

#rd_modes = RDModes(tl_data['c_bg'], tl_data['x_a'], tl_data['z_a'],
rd_modes = RDModes(tl_data['c_total'], tl_data['x_a'], tl_data['z_a'],
                    cf.fc, cf.z_src, c_bounds=cf.c_bounds)

# find first two modes
test_i = rd_modes.mode_number < 3
k_test = rd_modes.k_bg[test_i]

mn = np.arange(k_test.size - 1) + 1
k_diff = np.diff(np.real(k_test))
k_d_max = np.argmax(k_diff)
#k_d_max = 51
loop_len = -2 * pi / k_diff

fig, ax = plt.subplots()
ax.plot(loop_len, '.')

fig, ax = plt.subplots()
#ax.plot(rd_modes.psi_bg[k_d_max + np.arange(-1, 2), :].T, rd_modes.z_a)
ax.plot(rd_modes.psi_bg[k_d_max + np.arange(-2, 3), :].T, rd_modes.z_a)
#ax.plot(rd_modes.psi_bg[k_d_max + np.arange(-3, 4), :].T, rd_modes.z_a)
ax.set_ylim(150, 0)
ax.set_title(f'{x_start}')

#[plot(xs) for xs in np.arange(20) * 10]

1/0

peaks = find_peaks(k_diff)[0]
p_i = np.argsort(k_diff[peaks])[-3:]

test_i = np.where(k_diff > k_diff[peaks[p_i[0]]])[0]
mode_split = np.where(np.diff(test_i) > 1)[0]

if mode_split.size > 1:
    1/0

m1 = test_i[:mode_split[0] + 1]
m2 = test_i[mode_split[0] + 1:]

z_i = tl_data['z_a'] < z_int
z_a = tl_data['z_a'][z_i]
x_a = rd_modes.r_plot + rd_modes.x_start
dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

# make min 3 modes
m1 = peaks[p_i[-1]] + np.arange(-1, 2)
m2 = peaks[p_i[-2]] + np.arange(-2, 3)

mode_indicies = [m1, m2]

