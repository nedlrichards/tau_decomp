import matplotlib
#matplotlib.use('Agg')

import numpy as np
from math import pi
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from copy import copy
from scipy.interpolate import UnivariateSpline
from scipy.signal import stft, blackmanharris, iirfilter, sosfreqz, sosfiltfilt
from scipy.fft import fftshift
from scipy.stats import linregress

import os

from src import RDModes, Config

plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')
bbox = dict(boxstyle='round', fc='w')

fc = 400
cf = Config(fc)

def integrate_pressure(rd_modes, amplitudes, z_a):

    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
    p_slice = rd_modes.synthesize_pressure(amplitudes, z_a)
    p_scaled = p_slice * np.sqrt(rd_modes.r_plot[:, None])


    energy = np.sum(np.abs(p_scaled) ** 2, axis=-1) * dz
    return energy

def plot(x_start, field):
    run_number = int(x_start)
    tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                    f'tl_section_{run_number}.npz'))

    rd_modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds)


    z_a = tl_data['z_a'][tl_data['z_a'] < 120.]
    x_a = rd_modes.r_plot + rd_modes.x_start

    m1_amps = tl_data[field_type + '_mode_amps'].copy()
    m1_amps[:, rd_modes.mode_number != 0] = 0
    m1_en = integrate_pressure(rd_modes, m1_amps, z_a)

    m2_amps = tl_data[field_type + '_mode_amps'].copy()
    m2_amps[:, rd_modes.mode_number != 1] = 0
    m2_en = integrate_pressure(rd_modes, m2_amps, z_a)

    m_total_en = integrate_pressure(rd_modes, tl_data[field_type + '_mode_amps'], z_a)

    fig, ax = plt.subplots(2, 1, sharex=True)
    p_plot = tl_data['p_' + field_type]

    cm = ax[0].pcolormesh(tl_data['rplot'] / 1e3, tl_data['zplot'],
                            20 * np.log10(np.abs(p_plot)).T,
                            cmap=cmap, vmax=-50, vmin=-90, rasterized=True)

    cax = fig.add_axes([.83, .50, .02, .4])
    cb = fig.colorbar(cm, cax=cax)
    #cb.set_label('Acoustic pressure (dB re 1m)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_ylim(150, 0)

    ax[1].plot(x_a / 1e3, 100 * m2_en / m_total_en, 'k')


    #ax[1].plot(x_a / 1e3, 10 * np.log10(m1_en), label="Mode 1")
    #ax[1].plot(x_a / 1e3, 10 * np.log10(m2_en), label="Mode 2")
    #ax[1].plot(x_a / 1e3, 10 * np.log10(m_total_en), 'k', label='Total')
    #ax[1].legend()
    #ax[1].set_ylim(-35, -5)
    ax[1].set_ylim(0, 70)
    ax[1].set_ylabel('Integrated Energy')
    ax[1].set_xlabel('Position, $x$ (m)')

    pos = ax[0].get_position()
    pos.x0 += 0.06
    pos.x1 -= 0.06
    pos.y0 += 0.04
    pos.y1 += 0.07
    ax[0].set_position(pos)

    pos = cb.ax.get_position()
    pos.x0 += 0.03
    pos.x1 += 0.03
    pos.y0 += 0.04
    pos.y1 += 0.07
    cb.ax.set_position(pos)

    pos = ax[1].get_position()
    pos.x0 += 0.06
    pos.x1 -= 0.06
    pos.y0 += 0.04
    pos.y1 += 0.07
    ax[1].set_position(pos)

    fig.savefig('reports/figures/tl_v_mode_en/' + f'sec_{run_number}.png', dpi=300)
    plt.close(fig)

positions = np.arange(0, 970, 10)
field_type = 'spice'

for xs in positions:
    plot(xs, field_type)
