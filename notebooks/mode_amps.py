import numpy as np
from math import pi
import matplotlib.pyplot as plt


import os

from src import Config, list_tl_files, RDModes, section_cfield

#plt.ion()

fc = 400
source_depth = 'shallow'
tl_files = list_tl_files(fc, source_depth=source_depth)

cf = Config(fc=fc, source_depth=source_depth)

#field_type = 'spice'
field_type = 'total'
#field_type = 'tilt'


def comp_plot(tl_file):

    fields = np.load('data/processed/inputed_decomp.npz')
    x_a = fields['x_a']
    z_a = fields['z_a']
    c_field = fields['c_' + field_type]

    tl_data = np.load(tl_file)
    rplot = tl_data['rplot']
    zplot = tl_data['zplot']

    x_sec, c_field = section_cfield(tl_data['xs'], x_a, c_field)

    rd_modes = RDModes(c_field, tl_data['x_a'], tl_data['z_a'], cf)
    r_modes = (rd_modes.r_plot + tl_data['xs']) / 1e3
    #mode_amps = rd_modes.couple_cn()
    #p_modes = rd_modes.synthesize_pressure(mode_amps, zplot)

    ll = -2 * pi / (np.diff(rd_modes.k_bg))
    p_i = np.argmax(ll)
    m_range = (-50, 150)

    cm_i = np.arange(p_i + m_range[0], p_i + m_range[1])
    cm_i = cm_i[cm_i >= 0]

    rd_trunc = RDModes(c_field, tl_data['x_a'], tl_data['z_a'], cf,
                    psi_k_bg=(rd_modes.psi_bg[cm_i, :], rd_modes.k_bg[cm_i]))

    trunc_mode_amps = rd_trunc.couple_cn()
    p_trunc = rd_trunc.synthesize_pressure(trunc_mode_amps, zplot)

    z_line = 50
    plot_i = np.argmin(np.abs(zplot - z_line))

    #fig, ax = plt.subplots()
    #ax.plot(rplot / 1e3, 20 * np.log10(np.abs(tl_data['p_' + field_type]))[:, plot_i])
    #ax.plot(rplot / 1e3, 20 * np.log10(np.abs(p_modes))[:, plot_i])
    #ax.plot(rplot / 1e3, 20 * np.log10(np.abs(p_trunc))[:, plot_i])

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6.5, 3))

    cm = ax[0].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(tl_data['p_' + field_type])).T,
                            cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

    #20 * np.log10(np.abs(p_modes)).T,
    cm = ax[1].pcolormesh(rplot / 1e3, zplot,
                            20 * np.log10(np.abs(p_trunc)).T,
                            cmap=cf.cmap, vmax=-50, vmin=-90, rasterized=True)

    ax[0].set_ylim(150, 0)

    savename = tl_file.split('/')[-1].split('.')[0]

    fig.savefig(os.path.join('notebooks/figures/tl_comp', savename))
    plt.close(fig)

[comp_plot(tl) for tl in tl_files]
#comp_plot(tl_files[9])
