import matplotlib
#matplotlib.use('Agg')

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

x_start = 420
field_type = 'spice'

run_number = int(x_start)
tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                    f'tl_section_{run_number}.npz'))
cf = Config(fc)

rd_modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                    cf.fc, cf.z_src, c_bounds=cf.c_bounds)

z_i = tl_data['z_a'] < z_int
z_a = tl_data['z_a'][z_i]
x_a = rd_modes.r_plot + rd_modes.x_start

dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

def pressure_synthesis(mode_shapes, mode_k, mode_amps, z_rcr):
    """Compute spreading compensated pressure"""
    r_synth = rd_modes.r_plot
    mode_k = np.array(mode_k, ndmin=1)

    if len(mode_amps.shape) == 1:
        mode_amps = mode_amps[:, None]

    psi_ier = interp1d(z_a, mode_shapes)
    psi_rcr = psi_ier(z_rcr)

    if len(psi_rcr.shape) == len(mode_shapes.shape) - 1:
        k = mode_k[None, :]
        r = r_synth[:, None]
        psi = psi_rcr
        amps = mode_amps
    else:
        k = mode_k[None, None, :]
        r = r_synth[:, None, None]
        psi = (np.array(psi_rcr, ndmin=2).T)[None, :, :]
        amps = mode_amps[:, None, :]

    pressure = amps * psi * np.exp(1j * k * r)

    pressure = pressure.sum(axis=-1)
    return pressure

def truncated_synthesis(indicies, z_rcr):
    """Common synthesis procedure for a restricted number of modes"""
    amps = tl_data[field_type + '_mode_amps'][:, indicies]
    k = rd_modes.k_bg[indicies]
    modes = rd_modes.psi_bg[indicies, :][:, z_i]

    return pressure_synthesis(modes, k, amps, z_rcr)

def merged_synthesis(indicies, z_rcr):
    """merge multiple modes into a single mode"""

    modes = rd_modes.psi_bg[indicies, :][:, z_i]
    amps = tl_data[field_type + '_mode_amps'][:, indicies]
    k = rd_modes.k_bg[indicies]

    merged_mode = np.mean(modes, axis=0)

    # normalize to mode maxima
    amp_norm = np.max(np.abs(modes), axis=1) / np.max(np.abs(merged_mode))

    scaled_amps = amps * amp_norm[None, :] \
                * np.exp(1j * k * rd_modes.r_plot[:, None])
    merged_amp = scaled_amps.sum(axis=1)

    return merged_mode, pressure_synthesis(merged_mode, 0, merged_amp, z_rcr)

# mode 1 synthesis
#z_test = 32
#z_test = 50
#z_test = 65

#reference_p = truncated_synthesis(np.arange(rd_modes.mode_number.size), z_a)
reference_p = rd_modes.synthesize_pressure(tl_data[field_type + '_mode_amps'],
                                            z_a)
#reference_p *= np.sqrt(rd_modes.r_plot)[:, None]

m1_ind = np.where(rd_modes.mode_number == 0)[0]

mode_indicies = [np.arange(71,73), np.arange(73,78), np.arange(78,83), np.arange(83,88)]

partial_p = []
merged_modes = []
merged_p = []

for ind in mode_indicies:
    partial_p.append(truncated_synthesis(ind, z_a))
    m_mode, m_p = merged_synthesis(ind, z_a)
    merged_modes.append(m_mode)
    merged_p.append(m_p)

plot_ind = 2
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
ax[0].pcolormesh(x_a / 1e3, z_a, 20 * np.log10(np.abs(partial_p[plot_ind])).T,
              vmax=-40, vmin=-60, cmap=cmap)
ax[1].pcolormesh(x_a / 1e3, z_a, 20 * np.log10(np.abs(merged_p[plot_ind])).T,
              vmax=-40, vmin=-60, cmap=cmap)
ax[0].set_ylim(150, 0)


plot_ind = [0, 1, 2]
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6.5, 6))
ax[0].pcolormesh(x_a / 1e3, z_a, 20 * np.log10(np.abs(reference_p)).T,
                 vmax=-50, vmin=-90, cmap=cmap)
#ax[1].pcolormesh(tl_data['rplot'] / 1e3, tl_data['zplot'],
                 #20 * np.log10(np.abs(tl_data['p_' + field_type])).T,
                 #vmax=-50, vmin=-90, cmap=cmap)

p_temp = np.sum(np.array([partial_p[p_i] for p_i in plot_ind]), axis=0)
ax[1].pcolormesh(x_a / 1e3, z_a, 20 * np.log10(np.abs(p_temp)).T,
                 vmax=-25, vmin=-60, cmap=cmap)
p_temp = np.sum(np.array([merged_p[p_i] for p_i in plot_ind]), axis=0)
ax[2].pcolormesh(x_a / 1e3, z_a, 20 * np.log10(np.abs(p_temp)).T,
                 vmax=-25, vmin=-60, cmap=cmap)
ax[0].set_ylim(150, 0)

