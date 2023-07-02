import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

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
range_bounds = np.array((7.5e3, 47.5e3))
ml_tl_rb = (7.5e3, 37.0e3)
deep_rb = (7.5e3, 40.0e3)
savedir = 'reports/jasa/tex'
cf = Config()

load_dir = 'data/processed/'

shallow_int_400 = np.load(join(load_dir, 'int_eng_shallow_400.npz'))
shallow_int_1000 = np.load(join(load_dir, 'int_eng_shallow_1000.npz'))

r_a = shallow_int_400['r_a']
block_i = shallow_int_400['block_i'] & shallow_int_1000['block_i']

deep_int_400 = np.load(join(load_dir, 'int_eng_deep_400.npz'))
deep_int_1000 = np.load(join(load_dir, 'int_eng_deep_1000.npz'))

def z_nn(v):
    """Format negative number close to zero as 0.0"""
    return 0.0 if round(v,1) == 0 else v

def sparkline(ax, r_line_a, lines, stats, dy, title, ylim=(-10, 5), dx=0):
    locator = FixedLocator(ylim)
    ax.yaxis.set_major_locator(locator)

    r_stat_a = stats['r_a']

    ax.plot(r_line_a / 1e3, lines.T, linewidth=0.5, color='0.7')

    # mark lines crossing past ylim
    cross_i = np.argmax(lines < ylim[0], axis=-1)
    cross_r = r_line_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[0]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')

    # mark lines crossing past ylim
    cross_i = np.argmax(lines > ylim[1], axis=-1)
    cross_r = r_line_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[1]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')


    ax.plot(r_stat_a / 1e3, stats['mean'], linewidth=1.5, color='k')
    ax.plot(r_stat_a / 1e3, (stats['mean'] + stats['rms']), linewidth=0.75, linestyle='--', color='k')
    ax.plot(r_stat_a / 1e3, (stats['mean'] - stats['rms']), linewidth=0.75, linestyle='--', color='k')

    #m_ra, mean = rgs(r_a, 'mean', stats, range_bounds=rb)
    #_, rms = rgs(r_a, 'rms', stats, range_bounds=rb)

    lin_rgs = stats["mean_rgs"]
    mean = lin_rgs.intercept + range_bounds * lin_rgs.slope
    lin_rgs = stats['rms_rgs']
    rms = lin_rgs.intercept + range_bounds * lin_rgs.slope

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
    else:
        lv = rf"&\bf{{{z_nn(mean[0]): 3.1f}}}\\"

    if stats['mean'][0] < 0:
        left_str += lv
    else:
        left_str += rf"  \ \, \ " + lv

    if stats['mean'][-1] < 0:
        rght_str += rf"&\bf{{{z_nn(mean[-1]): 3.1f}}}\\"
    else:
        rght_str += rf"\ \, \ & \bf{{{z_nn(mean[-1]): 3.1f}}}\\"

    left_str += rf"& \ \, \ \pm {z_nn(rms[0]):3.1f}"
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


def plot_sparks(r_a, lines_400, lines_1000, ylim=(-10, 5), stat_range=(10e3, 40e3)):
    """Four by 2 plot that does not correct for blocking"""
    stat_range = np.array(stat_range)
    demean_400 = lines_400[1:] - lines_400[0]
    stats_400 = [field_stats(r_a, v, range_bounds=stat_range) for v in demean_400]

    demean_1000 = lines_1000[1:] - lines_1000[0]
    stats_1000 = [field_stats(r_a, v, range_bounds=stat_range) for v in demean_1000]
    space_str = "\hspace{2.0em}"
    space_str_1 = "\hspace{3.4em}"

    bg_i = cf.field_types.index('bg')
    tilt_i = cf.field_types.index('tilt')
    spice_i = cf.field_types.index('spice')
    total_i = cf.field_types.index('total')


    fig, axes = plt.subplots(7, 2, figsize=(cf.jasa_2clm, 3.75))
    #r_i = stats_400[bg_i]['r_i']
    r_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])
    r_line_a = r_a[r_i]

    ax = axes[0, 0]
    col_str = f'Type {space_str_1} 400 Hz, dB re RI BG  {space_str} {range_bounds[0]/1e3:.1f} km  {space_str} {range_bounds[1]/1e3} km'
    ax.text(-0.48, 2.3, col_str, transform=ax.transAxes)
    ax.text(-0.45, 1.5, 'Complete', transform=ax.transAxes, clip_on=False, bbox=cf.bbox, fontsize=8)
    ax.plot([-0.46, 4.17], [1.65, 1.65], transform=ax.transAxes, clip_on=False, linewidth=0.75, color='0.8')
    sparkline(ax, r_line_a, demean_400[bg_i][:, r_i], stats_400[bg_i], -0.04, 'BG', ylim=ylim)
    ax.set_yticks(ylim)
    ax.set_yticklabels(ylim, fontdict={'fontsize':8})

    ax = axes[1, 0]
    sparkline(ax, r_line_a, demean_400[tilt_i][:, r_i], stats_400[tilt_i], -0.04, 'Tilt', ylim=ylim)

    ax = axes[2, 0]
    sparkline(ax, r_line_a, demean_400[spice_i][:, r_i], stats_400[spice_i], -0.04, 'Spice', ylim=ylim)

    ax = axes[3, 0]
    sparkline(ax, r_line_a, demean_400[total_i][:, r_i], stats_400[total_i], -0.04, 'Obs.', ylim=ylim)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    ax = axes[0, 1]
    col_str = f'1 kHz, dB re RI BG  {space_str} {range_bounds[0]/1e3:.1f} km  {space_str} {range_bounds[1]/1e3} km'
    ax.text(0.10, 2.3, col_str, transform=ax.transAxes)

    sparkline(ax, r_line_a, demean_1000[bg_i][:, r_i], stats_1000[bg_i], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[1, 1]
    sparkline(ax, r_line_a, demean_1000[tilt_i][:, r_i], stats_1000[tilt_i], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[2, 1]
    sparkline(ax, r_line_a, demean_1000[spice_i][:, r_i], stats_1000[spice_i], -0.04, '', dx=0.04, ylim=ylim)

    ax = axes[3, 1]
    sparkline(ax, r_line_a, demean_1000[total_i][:, r_i], stats_1000[total_i], -0.04, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    ax = axes[4, 0]
    ax.plot([-0.46, 4.17], [1.5, 1.5], transform=ax.transAxes, clip_on=False, linewidth=0.75, color='0.8')
    ax.text(-0.45, 1.5, 'W/O Blocking', transform=ax.transAxes, clip_on=False, bbox=cf.bbox, fontsize=8)

    flt_eng = demean_400[tilt_i][block_i[tilt_i], :].copy()
    flt_tilt_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_tilt_stats, -0.10, 'Tilt', ylim=ylim)
    ax.set_yticks(ylim)
    ax.set_yticklabels(ylim, fontdict={'fontsize':8})


    ax = axes[5, 0]

    flt_eng = demean_400[spice_i][block_i[spice_i], :].copy()
    flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_spice_stats, -0.10, 'Spice', ylim=ylim)

    ax = axes[6, 0]

    flt_eng = demean_400[total_i][block_i[total_i], :].copy()
    flt_total_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_total_stats, -0.10, 'Obs.', ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    ax = axes[4, 1]
    flt_eng = demean_1000[tilt_i][block_i[tilt_i], :].copy()
    flt_tilt_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_tilt_stats, -0.10, '', dx=0.04, ylim=ylim)

    ax = axes[5, 1]

    flt_eng = demean_1000[spice_i][block_i[spice_i], :].copy()
    flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_spice_stats, -0.10, '', dx=0.04, ylim=ylim)

    ax = axes[6, 1]

    flt_eng = demean_1000[total_i][block_i[total_i], :].copy()
    flt_total_stats = field_stats(r_a, flt_eng, range_bounds=stat_range)
    sparkline(ax, r_line_a, flt_eng[:, r_i], flt_total_stats, -0.10, '', dx=0.04, ylim=ylim)

    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
    ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
    ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
    ax.tick_params(bottom=True)
    ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

    return fig, axes

fig, axes = plot_sparks(r_a, shallow_int_400['ml_ml'], shallow_int_1000['ml_ml'],
                        ylim=(-10, 5), stat_range=range_bounds)
fig.savefig(join(savedir, 'figure_15.pdf'), dpi=300)

fig, axes = plot_sparks(r_a, shallow_int_400['ml_proj'], shallow_int_1000['ml_proj'],
                        ylim=(-20, 5), stat_range=range_bounds)
fig.savefig(join(savedir, f'figure_16.pdf'), dpi=300)

fig, axes = plot_sparks(r_a, deep_int_400['ml_ml'], deep_int_1000['ml_ml'],
                        ylim=(-20, 20), stat_range=deep_rb)
fig.savefig(join(savedir, 'figure_18.pdf'), dpi=300)


# transition layer
# use a 2nd order fit of the mean for fit
rb_i = (r_a >= ml_tl_rb[0]) & (r_a <= ml_tl_rb[1])

test_400 = shallow_int_400['ml_tl'].copy()
bg = test_400[0]
bg_fit_400 = []
for line in bg:
    fit = np.polynomial.polynomial.polyfit(r_a[rb_i], line[rb_i], 1)
    bg_fit_400.append(np.polynomial.polynomial.polyval(r_a, fit))
bg_fit_400 = np.array(bg_fit_400)
test_400[0] = bg_fit_400

shallow_modelled = np.load(join(load_dir, 'tl_eng_model_400.npz'))['ml_pred']
#test_400[1:] = shallow_modelled

test_1000 = shallow_int_1000['ml_tl'].copy()
bg = test_1000[0]
bg_fit_1000 = []
for line in bg:
    fit = np.polynomial.polynomial.polyfit(r_a[rb_i], line[rb_i], 2)
    bg_fit_1000.append(np.polynomial.polynomial.polyval(r_a, fit))
bg_fit_1000 = np.array(bg_fit_1000)
test_1000[0] = bg_fit_1000

shallow_modelled = np.load(join(load_dir, 'tl_eng_model_1000.npz'))['ml_pred']
#test_1000[1:] = shallow_modelled

fig, axes = plot_sparks(r_a, test_400, test_1000, ylim=(-20, 25), stat_range=ml_tl_rb)
fig.savefig(join(savedir, f'figure_17.pdf'), dpi=300)
