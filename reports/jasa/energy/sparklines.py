import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import EngProc, Config, list_tl_files, MLEnergy

import matplotlib as mpl
custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}", # for the align enivironment
    }

mpl.rcParams.update(custom_preamble)

plt.style.use('elr')
plt.ion()

source_depth="shallow"
range_bounds = (7.5e3, 47.5e3)
fc = 1000
savedir = 'reports/jasa/figures'
cf = Config()

def mode_eng(fc, mode_num):
    """Project mode onto pressure field to compute amplitude"""
    dyn = {'bg':[], 'tilt':[], 'spice':[], 'total':[]}
    e_ri = []

    tl_files = list_tl_files(fc=fc)
    for tl in tl_files:
        eng_mode = MLEnergy(tl)
        eng_bg = eng_mode.background_diffraction('bg')
        e_ri.append(10 * np.log10(eng_bg * eng_mode.r_a))

        for ft in cf.field_types:
            proj_amp, proj_scale = eng_mode.proj_mode(ft, mode_num=mode_num)
            proj_eng = np.abs(proj_amp) ** 2 / proj_scale
            dyn[ft].append(10 * np.log10(proj_eng))

    lines = {'bg':np.array(dyn['bg']) - e_ri,
             'tilt':np.array(dyn['tilt']) - e_ri,
             'spice':np.array(dyn['spice']) - e_ri,
             'total':np.array(dyn['total']) - e_ri}

    return lines

eng_1 = EngProc(Config(fc=1000, source_depth=source_depth), fields=['bg'])
e_ri_1 = eng_1.diffraction_bg()
dyn_1 = eng_1.dynamic_energy()

eng_4 = EngProc(Config(fc=400, source_depth=source_depth), fields=['bg'])
e_ri_4 = eng_4.diffraction_bg()
dyn_4 = eng_4.dynamic_energy()

lines_eng_4 = {'bg':dyn_4['bg'] - e_ri_4,
               'tilt':dyn_4['tilt'] - e_ri_4,
               'spice':dyn_4['spice'] - e_ri_4,
               'total':dyn_4['total'] - e_ri_4}
lines_eng_1 = {'bg':dyn_1['bg'] - e_ri_1,
               'tilt':dyn_1['tilt'] - e_ri_1,
               'spice':dyn_1['spice'] - e_ri_1,
               'total':dyn_1['total'] - e_ri_1}

lines_proj_4 = mode_eng(400, 1)
lines_proj_1 = mode_eng(1e3, 2)

# Blocking features computed at 400 Hz
dyn = np.array([dyn_4[fld] for fld in cf.field_types])
bg_i = eng_4.blocking_feature(dyn, e_ri_4, range_bounds=range_bounds)
block_i = bg_i < 3
# Blocking features at 1 kHz
dyn = np.array([dyn_1[fld] for fld in cf.field_types])
bg_i = eng_1.blocking_feature(dyn, e_ri_1, range_bounds=range_bounds)
block_i &= bg_i < 3  # blocking features at either freqeuncy count

def sparkline(ax, lines, stats, dy, title, ylim=(-10, 5), dx=0):
    r_a = stats['r_a']

    ax.plot(r_a / 1e3, lines.T, linewidth=0.5, color='0.7')

    # mark lines crossing past ylim
    cross_i = np.argmax(lines < ylim[0], axis=-1)
    cross_r = r_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[0]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')

    ax.plot(r_a / 1e3, stats['mean'], linewidth=1.5, color='k')
    ax.plot(r_a / 1e3, (stats['mean'] + stats['rms']), linewidth=0.75, linestyle='--', color='k')
    ax.plot(r_a / 1e3, (stats['mean'] - stats['rms']), linewidth=0.75, linestyle='--', color='k')

    m_ra, mean = eng_4.rgs('mean', stats, range_bounds=range_bounds)
    _, rms = eng_4.rgs('rms', stats, range_bounds=range_bounds)

    #ax.plot(m_ra / 1e3, mean, linewidth=1.5, color='k')
    #ax.plot(m_ra / 1e3, mean + rms, linewidth=0.75, linestyle='--', color='k')
    #ax.plot(m_ra / 1e3, mean - rms, linewidth=0.75, linestyle='--', color='k')

    #ax.plot(r_a / 1e3, stats['15th'][inds], linewidth=1, color='r', linestyle='--')
    #ax.plot(r_a / 1e3, stats['85th'][inds], linewidth=1, color='r', linestyle='--')


    pos = ax.get_position()
    pos.x0 += -0.02 + dx
    pos.x1 += -0.16 + dx
    pos.y0 += 0.05 + dy
    pos.y1 += 0.03 + dy
    ax.set_position(pos)

    left_str = r"\begin{align}"
    rght_str = r"\begin{align}"

    if title == 'BG':
        lv = rf"&\bf{{{mean[0]: 3.1f} \ \textrm{{(dB)}}}}\\"
        #lv = rf"&\bf{{{stats['mean'][0]: 3.1f} \ \textrm{{(dB)}}}}\\"
    else:
        lv = rf"&\bf{{{mean[0]: 3.1f}}}\\"
        #lv = rf"&\bf{{{stats['mean'][0]: 3.1f}}}\\"

    if stats['mean'][0] < 0:
        left_str += lv
    else:
        left_str += rf"  \ \, \ " + lv

    if stats['mean'][-1] < 0:
        rght_str += rf"&\bf{{{mean[-1]: 3.1f}}}\\"
        #rght_str += rf"&\bf{{{stats['mean'][-1]: 3.1f}}}\\"
    else:
        rght_str += rf"\ \, \ & \bf{{{mean[-1]: 3.1f}}}\\"
        #rght_str += rf"\ \, \ & \bf{{{stats['mean'][-1]: 3.1f}}}\\"

    #left_str += rf"& \ \, \ \pm {stats['rms'][0]:3.1f}"
    left_str += rf"& \ \, \ \pm {rms[0]:3.1f}"
    #rght_str += rf"& \ \, \ \pm {stats['rms'][-1]:3.1f}"
    rght_str += rf"& \ \, \ \pm {rms[-1]:3.1f}"
    left_str += r"\end{align}"
    rght_str += r"\end{align}"

    ax.text(1.10, 0.7, left_str, transform=ax.transAxes)
    ax.text(1.60, 0.7, rght_str, transform=ax.transAxes)
    ax.text(-0.46, 0.6, title, transform=ax.transAxes)

    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.tick_params(bottom=False, right=True)
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])
    ax.set_xlim(range_bounds[0] / 1e3 - 1, range_bounds[1] / 1e3 + 1)
    ax.set_ylim(ylim[0], ylim[1])


def plot_w_blocking(energy_400, lines_400, energy_1000, lines_1000, ylim=(-10, 5)):
    """Four by 2 plot that does not correct for blocking"""
    stats_400 = {i:energy_400.field_stats(v, range_bounds=range_bounds) for i, v in lines_400.items()}
    stats_1000 = {i:energy_1000.field_stats(v, range_bounds=range_bounds) for i, v in lines_1000.items()}

    fig, axes = plt.subplots(4, 2, figsize=(cf.jasa_2clm, 2.25))
    r_i = stats_400['bg']['r_i']
    ax = axes[0, 0]
    sparkline(ax, lines_400['bg'][:, r_i], stats_400['bg'], -0.04, 'BG', ylim=ylim)
    col_str = 'Type \hspace{3.4em} 400 Hz, dB re RI BG  \hspace{1.7em} 7.5 km  \hspace{2.0em} 47.5 km'
    ax.text(-0.48, 1.4, col_str, transform=ax.transAxes)
    ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

    ax = axes[1, 0]
    sparkline(ax, lines_400['tilt'][:, r_i], stats_400['tilt'], -0.04, 'Tilt', ylim=ylim)

    ax = axes[2, 0]
    sparkline(ax, lines_400['spice'][:, r_i], stats_400['spice'], -0.04, 'Spice', ylim=ylim)

    ax = axes[3, 0]
    sparkline(ax, lines_400['total'][:, r_i], stats_400['total'], -0.04, 'Obs.', ylim=ylim)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)

    ax = axes[0, 1]
    sparkline(ax, lines_1000['bg'][:, r_i], stats_1000['bg'], -0.04, '', dx=0.04, ylim=ylim)
    col_str = '1 kHz, dB re RI BG  \hspace{2.0em} 7.5 km  \hspace{2.0em} 47.5 km'
    ax.text(0.10, 1.4, col_str, transform=ax.transAxes)

    ax = axes[1, 1]
    sparkline(ax, lines_1000['tilt'][:, r_i], stats_1000['tilt'], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[2, 1]
    sparkline(ax, lines_1000['spice'][:, r_i], stats_1000['spice'], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[3, 1]
    sparkline(ax, lines_1000['total'][:, r_i], stats_1000['total'], -0.04, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    return fig, axes


def plot_w_o_blocking(energy_400, lines_400, energy_1000, lines_1000, blocking_inds, ylim=(-10, 5)):
    """three by 2 plot that does correct for blocking"""
    stats_400 = {i:energy_400.field_stats(v, range_bounds=range_bounds) for i, v in lines_400.items()}
    stats_1000 = {i:energy_1000.field_stats(v, range_bounds=range_bounds) for i, v in lines_1000.items()}

    inds = stats_400['bg']['r_i']
    fig, axes = plt.subplots(3, 2, figsize=(cf.jasa_2clm, 2.00))

    ax = axes[0, 0]
    flt_eng = lines_400['tilt'][block_i[cf.field_types.index('tilt')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_tilt_stats = energy_400.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_tilt_stats, -0.04, 'Tilt', ylim=ylim)

    ax.text(-0.48, 1.3, 'Type \hspace{3.4em} 400 Hz, dB re RI BG  \hspace{1.7em} 7.5 km  \hspace{2.0em} 47.5 km',
            transform=ax.transAxes)
    ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

    ax = axes[1, 0]

    flt_eng = lines_400['spice'][block_i[cf.field_types.index('spice')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_spice_stats = energy_400.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_spice_stats, -0.02, 'Spice', ylim=ylim)

    ax = axes[2, 0]

    flt_eng = lines_400['total'][block_i[cf.field_types.index('total')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_total_stats = energy_400.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_total_stats, -0.00, 'Obs.', ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    ax = axes[0, 1]
    flt_eng = lines_1000['tilt'][block_i[cf.field_types.index('tilt')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_tilt_stats = energy_1000.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_tilt_stats, -0.04, '', dx=0.04, ylim=ylim)
    ax.text(0.10, 1.3, '1 kHz, dB re RI BG  \hspace{2.0em} 7.5 km  \hspace{2.0em} 47.5 km',
            transform=ax.transAxes)

    ax = axes[1, 1]

    flt_eng = lines_1000['spice'][block_i[cf.field_types.index('spice')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_spice_stats = energy_1000.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_spice_stats, -0.02, '', dx=0.04, ylim=ylim)

    ax = axes[2, 1]

    flt_eng = lines_1000['total'][block_i[cf.field_types.index('total')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_total_stats = energy_1000.field_stats(flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_total_stats, -0.00, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    return fig, axes


fig, axes = plot_w_blocking(eng_4, lines_eng_4, eng_1, lines_eng_1)
fig.savefig(join(savedir, f'shallow_blocking.png'), dpi=300)
fig, axes = plot_w_blocking(eng_4, lines_proj_4, eng_1, lines_proj_1, ylim=(-20, 5))
fig.savefig(join(savedir, f'shallow_proj_blocking.png'), dpi=300)


fig, axes = plot_w_o_blocking(eng_4, lines_eng_4, eng_1, lines_eng_1, block_i)
fig.savefig(join(savedir, f'shallow_no_blocking.png'), dpi=300)
fig, axes = plot_w_o_blocking(eng_4, lines_proj_4, eng_1, lines_proj_1, block_i, ylim=(-20, 5))
fig.savefig(join(savedir, f'shallow_proj_no_blocking.png'), dpi=300)



