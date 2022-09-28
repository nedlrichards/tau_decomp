import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats, rgs

import matplotlib as mpl
custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}", # for the align enivironment
    }

mpl.rcParams.update(custom_preamble)

plt.style.use('elr')
plt.ion()

#source_depth="shallow"
source_depth="deep"
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

def compute_statistics(fc):
    """statistics used in sparkline plots"""
    eng = EngProc(Config(fc=fc, source_depth=source_depth), fields=['bg'])
    e_ri = eng.diffraction_bg()
    dyn = eng.dynamic_energy()
    lines_eng = {'bg':dyn['bg'] - e_ri,
                'tilt':dyn['tilt'] - e_ri,
                'spice':dyn['spice'] - e_ri,
                'total':dyn['total'] - e_ri}
    # Blocking features computed at 400 Hz
    d = np.array([dyn[fld] for fld in cf.field_types])
    bg_i = eng.blocking_feature(d, e_ri, range_bounds=range_bounds)
    block_i = bg_i < 3
    r_a = eng.r_a.copy()
    return lines_eng, block_i, r_a

lines_eng_4, block_i, r_a = compute_statistics(400)
lines_eng_1, tmp_i, _ = compute_statistics(1000)
block_i &= tmp_i
lines_proj_4 = mode_eng(400, 1)
lines_proj_1 = mode_eng(1e3, 2)

def z_nn(v):
    """Format negative number close to zero as 0.0"""
    return 0.0 if round(v,1) == 0 else v

def sparkline(ax, lines, stats, dy, title, ylim=(-10, 5), dx=0):
    locator = FixedLocator(ylim)
    ax.yaxis.set_major_locator(locator)

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

    m_ra, mean = rgs(r_a, 'mean', stats, range_bounds=range_bounds)
    _, rms = rgs(r_a, 'rms', stats, range_bounds=range_bounds)

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
        lv = rf"&\bf{{{z_nn(mean[0]): 3.1f} \ \textrm{{(dB)}}}}\\"
        #lv = rf"&\bf{{{stats['mean'][0]: 3.1f} \ \textrm{{(dB)}}}}\\"
    else:
        lv = rf"&\bf{{{z_nn(mean[0]): 3.1f}}}\\"
        #lv = rf"&\bf{{{stats['mean'][0]: 3.1f}}}\\"

    if stats['mean'][0] < 0:
        left_str += lv
    else:
        left_str += rf"  \ \, \ " + lv

    if stats['mean'][-1] < 0:
        rght_str += rf"&\bf{{{z_nn(mean[-1]): 3.1f}}}\\"
        #rght_str += rf"&\bf{{{stats['mean'][-1]: 3.1f}}}\\"
    else:
        rght_str += rf"\ \, \ & \bf{{{z_nn(mean[-1]): 3.1f}}}\\"
        #rght_str += rf"\ \, \ & \bf{{{stats['mean'][-1]: 3.1f}}}\\"

    #left_str += rf"& \ \, \ \pm {stats['rms'][0]:3.1f}"
    left_str += rf"& \ \, \ \pm {z_nn(rms[0]):3.1f}"
    #rght_str += rf"& \ \, \ \pm {stats['rms'][-1]:3.1f}"
    rght_str += rf"& \ \, \ \pm {z_nn(rms[-1]):3.1f}"
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


def plot_sparks(r_a, lines_400, lines_1000, ylim=(-10, 5)):
    """Four by 2 plot that does not correct for blocking"""
    stats_400 = {i:field_stats(r_a, v, range_bounds=range_bounds) for i, v in lines_400.items()}
    stats_1000 = {i:field_stats(r_a, v, range_bounds=range_bounds) for i, v in lines_1000.items()}

    fig, axes = plt.subplots(7, 2, figsize=(cf.jasa_2clm, 3.75))
    r_i = stats_400['bg']['r_i']

    ax = axes[0, 0]
    sparkline(ax, lines_400['bg'][:, r_i], stats_400['bg'], -0.04, 'BG', ylim=ylim)
    col_str = 'Type \hspace{3.4em} 400 Hz, dB re RI BG  \hspace{1.7em} 7.5 km  \hspace{2.0em} 47.5 km'
    ax.text(-0.48, 2.3, col_str, transform=ax.transAxes)
    #ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})
    ax.set_yticklabels(ylim, fontdict={'fontsize':8})
    ax.text(-0.45, 1.5, 'Complete', transform=ax.transAxes, clip_on=False, bbox=cf.bbox, fontsize=8)

    ax.plot([-0.46, 4.17], [1.65, 1.65], transform=ax.transAxes, clip_on=False, linewidth=0.75, color='0.8')

    ax = axes[1, 0]
    sparkline(ax, lines_400['tilt'][:, r_i], stats_400['tilt'], -0.04, 'Tilt', ylim=ylim)

    ax = axes[2, 0]
    sparkline(ax, lines_400['spice'][:, r_i], stats_400['spice'], -0.04, 'Spice', ylim=ylim)

    ax = axes[3, 0]
    sparkline(ax, lines_400['total'][:, r_i], stats_400['total'], -0.04, 'Obs.', ylim=ylim)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    #ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    #ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    #ax.tick_params(bottom=True)

    ax = axes[0, 1]
    sparkline(ax, lines_1000['bg'][:, r_i], stats_1000['bg'], -0.04, '', dx=0.04, ylim=ylim)
    col_str = '1 kHz, dB re RI BG  \hspace{2.0em} 7.5 km  \hspace{2.0em} 47.5 km'
    ax.text(0.10, 2.3, col_str, transform=ax.transAxes)

    ax = axes[1, 1]
    sparkline(ax, lines_1000['tilt'][:, r_i], stats_1000['tilt'], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[2, 1]
    sparkline(ax, lines_1000['spice'][:, r_i], stats_1000['spice'], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[3, 1]
    sparkline(ax, lines_1000['total'][:, r_i], stats_1000['total'], -0.04, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    #ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    #ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    #ax.tick_params(bottom=True)

    inds = stats_400['bg']['r_i']

    ax = axes[4, 0]
    flt_eng = lines_400['tilt'][block_i[cf.field_types.index('tilt')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_tilt_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_tilt_stats, -0.10, 'Tilt', ylim=ylim)

    ax.plot([-0.46, 4.17], [1.5, 1.5], transform=ax.transAxes, clip_on=False, linewidth=0.75, color='0.8')
    ax.text(-0.45, 1.5, 'W/O Blocking', transform=ax.transAxes, clip_on=False, bbox=cf.bbox, fontsize=8)
    #ax.text(-0.48, 1.3, 'Type \hspace{3.4em} 400 Hz, dB re RI BG  \hspace{1.7em} 7.5 km  \hspace{2.0em} 47.5 km',
            #transform=ax.transAxes)
    ax.set_yticklabels(ylim, fontdict={'fontsize':8})
    #ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

    ax = axes[5, 0]

    flt_eng = lines_400['spice'][block_i[cf.field_types.index('spice')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_spice_stats, -0.10, 'Spice', ylim=ylim)

    ax = axes[6, 0]

    flt_eng = lines_400['total'][block_i[cf.field_types.index('total')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_total_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_total_stats, -0.10, 'Obs.', ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    ax = axes[4, 1]
    flt_eng = lines_1000['tilt'][block_i[cf.field_types.index('tilt')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_tilt_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_tilt_stats, -0.10, '', dx=0.04, ylim=ylim)
    #ax.text(0.10, 1.3, '1 kHz, dB re RI BG  \hspace{2.0em} 7.5 km  \hspace{2.0em} 47.5 km',
            #transform=ax.transAxes)

    ax = axes[5, 1]

    flt_eng = lines_1000['spice'][block_i[cf.field_types.index('spice')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_spice_stats, -0.10, '', dx=0.04, ylim=ylim)

    ax = axes[6, 1]

    flt_eng = lines_1000['total'][block_i[cf.field_types.index('total')], :].copy()
    # remove 3 dB down series
    lossy = flt_eng[:, inds][:, 0] < -3
    flt_eng = np.delete(flt_eng, lossy, axis=0)
    flt_total_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)

    sparkline(ax, flt_eng[:, inds], flt_total_stats, -0.10, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    return fig, axes


if source_depth == 'deep':
    ylim = (-15, 15)
else:
    ylim = (-10, 5)

fig, axes = plot_sparks(r_a, lines_eng_4, lines_eng_1, ylim=ylim)
fig.savefig(join(savedir, source_depth + f'_eng.png'), dpi=300)

if source_depth == 'shallow':
    fig, axes = plot_sparks(r_a, lines_proj_4, lines_proj_1, ylim=(-20, 5))
    fig.savefig(join(savedir, f'shallow_eng_proj.png'), dpi=300)
