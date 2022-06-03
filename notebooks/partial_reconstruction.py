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

#plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')
bbox = dict(boxstyle='round', fc='w')

fc = 400
z_int = 150.
cf = Config(fc)

def merged_synthesis(indicies, z_rcr, amps, rd_modes):
    """merge multiple modes into a single mode"""

    modes = rd_modes.psi_bg[indicies, :]
    k = rd_modes.k_bg[indicies]

    merged_mode = np.mean(modes, axis=0)

    # normalize to mode maxima
    amp_norm = np.max(np.abs(modes), axis=1) / np.max(np.abs(merged_mode))

    scaled_amps = amps * amp_norm[None, :]
    scaled_amps *= np.exp(1j * k * rd_modes.r_plot[:, None])

    merged_amp = scaled_amps.sum(axis=1)

    return merged_mode, merged_amp


def plot_field(x_start, field_type, fig):

    run_number = int(x_start)
    tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                    f'xmission_{run_number:03}.npz'))

    rd_modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds)

    # find first two modes
    test_i = rd_modes.mode_number < 3
    k_test = rd_modes.k_bg[test_i]

    mn = np.arange(k_test.size - 1) + 1
    k_diff = np.diff(np.real(k_test))

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
    #mode_indicies = [np.array([peaks[p_i[-1]]]), np.array([peaks[p_i[-2]]])]

    merged_modes = []
    merged_amps = []
    amps_norm = []

    for ind in mode_indicies:

        amps = tl_data[field_type + '_mode_amps'][:, ind]
        m_mode, amp_mode = merged_synthesis(ind, z_a, amps, rd_modes)
        merged_modes.append(m_mode)
        merged_amps.append(amp_mode)
        p_1_i = find_peaks(np.abs(m_mode))[0][0]
        amps_norm.append(np.abs(m_mode[p_1_i]))

    reference_p = tl_data['p_' + field_type]
    merged_amps = np.array(merged_amps)
    amps_norm = np.array(amps_norm)
    psi_groups = [rd_modes.psi_bg[mode_indicies[0]][:, z_i],
                  rd_modes.psi_bg[mode_indicies[1]][:, z_i]]

    mosiac = """
             AA
             BB
             CD
             """

    ax = fig.subplot_mosaic(mosiac)
    ax['A'].pcolormesh(tl_data['rplot'] / 1e3, tl_data['zplot'],
                       20 * np.log10(np.abs(reference_p)).T,
                       vmax=-40, vmin=-90, cmap=cmap)
    ax['A'].set_ylim(120, 0)
    ax['A'].set_xlim([x_start, x_start + 60])
    ax['B'].set_xlim([x_start, x_start + 60])
    ax['A'].xaxis.set_ticklabels([])
    ax['A'].xaxis.set_ticks(x_start + np.array([1, 2, 3, 4, 5]) * 10)
    ax['B'].xaxis.set_ticks(x_start + np.array([1, 2, 3, 4, 5]) * 10)
    ax['B'].plot(x_a / 1e3, 20 * np.log10(np.abs(merged_amps * amps_norm[:, None])).T)
    ax['C'].plot(psi_groups[0].T / amps_norm[0], z_a)
    ax['C'].plot(merged_modes[0][z_i] / amps_norm[0], z_a, 'k')
    ax['D'].plot(psi_groups[1].T / amps_norm[1], z_a)
    ax['D'].plot(merged_modes[1][z_i] / amps_norm[1], z_a, 'k')
    ax['B'].set_ylim(-55, -20)
    ax['C'].set_ylim(130, 0)
    ax['D'].set_ylim(130, 0)
    ax['D'].yaxis.set_ticklabels([])

    ax['A'].set_ylim(150, 0)
    ax['A'].text(x_start + 4, -10, field_type)

    pos = ax['A'].get_position()
    pos.x0 += 0.10
    pos.x1 += 0.06
    pos.y0 += 0.06
    pos.y1 += 0.06
    ax['A'].set_position(pos)

    pos = ax['B'].get_position()
    pos.x0 += 0.10
    pos.x1 += 0.06
    pos.y0 += 0.06
    pos.y1 += 0.06
    ax['B'].set_position(pos)

    pos = ax['C'].get_position()
    pos.x0 += 0.06
    pos.x1 += 0.06
    pos.y0 -= 0.04
    pos.y1 += 0.00
    ax['C'].set_position(pos)

    pos = ax['D'].get_position()
    pos.x0 += 0.06
    pos.x1 += 0.06
    pos.y0 -= 0.04
    pos.y1 += 0.00
    ax['D'].set_position(pos)

def one_page(x_start):
    fig = plt.figure(figsize=(6.5, 9))
    subfigs = fig.subfigures(2, 2, wspace=0.5)
    plot_field(x_start, 'bg', subfigs[0, 0])
    plot_field(x_start, 'total', subfigs[1, 0])
    plot_field(x_start, 'tilt', subfigs[0, 1])
    plot_field(x_start, 'spice', subfigs[1, 1])
    fig.savefig('reports/figures/red_amp/' + f'red_amp_{x_start:03}km.png')
    plt.close(fig)

[one_page(xs * 10) for xs in np.arange(91)]
