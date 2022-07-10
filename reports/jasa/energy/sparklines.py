import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

from src import EngProc, Config
from plot_bg import plot_bg

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

cf = Config(fc=fc, source_depth=source_depth)
eng = EngProc(cf)

eng_ri = eng.diffraction_bg()
dynamic_eng = eng.dynamic_energy()

bg_stats = eng.field_stats(dynamic_eng['bg'] - eng_ri, range_bounds=range_bounds)
tilt_stats = eng.field_stats(dynamic_eng['tilt'] - eng_ri, range_bounds=range_bounds)
spice_stats = eng.field_stats(dynamic_eng['spice'] - eng_ri, range_bounds=range_bounds)
total_stats = eng.field_stats(dynamic_eng['total'] - eng_ri, range_bounds=range_bounds)

tick_off = 1

def sparkline(ax, lines, stats, dy, title, ylim=(-10, 5)):
    inds = (stats['r_a'] > range_bounds[0]) & (stats['r_a'] < range_bounds[1])
    r_a = stats['r_a'][inds]

    ax.plot(r_a / 1e3, lines.T, linewidth=0.5, color='0.7')

    # mark lines crossing past ylim
    cross_i = np.argmax(lines < -10, axis=-1)
    cross_r = r_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[0]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')

    ax.plot(r_a / 1e3, stats['mean'][inds], linewidth=1.5, color='k')
    ax.plot(r_a / 1e3, (stats['mean'] + stats['rms'])[inds], linewidth=0.75, linestyle='--', color='k')
    ax.plot(r_a / 1e3, (stats['mean'] - stats['rms'])[inds], linewidth=0.75, linestyle='--', color='k')
    #ax.plot(r_a / 1e3, stats['15th'][inds], linewidth=1, color='r', linestyle='--')
    #ax.plot(r_a / 1e3, stats['85th'][inds], linewidth=1, color='r', linestyle='--')


    pos = ax.get_position()
    pos.x0 += 0.10
    pos.x1 -= 0.36
    pos.y0 += 0.05 + dy
    pos.y1 += 0.03 + dy
    ax.set_position(pos)

    left_str = r"\begin{align}"
    rght_str = r"\begin{align}"

    if title == 'BG':
        lv = rf"&\bf{{{stats['mean'][0]: 3.1f} \ \textrm{{(dB)}}}}\\"
    else:
        lv = rf"&\bf{{{stats['mean'][0]: 3.1f}}}\\"

    if stats['mean'][0] < 0:
        left_str += lv
    else:
        left_str += rf"  \ \, \ " + lv

    if stats['mean'][-1] < 0:
        rght_str += rf"&\bf{{{stats['mean'][-1]: 3.1f}}}\\"
    else:
        rght_str += rf"\ \, \ & \bf{{{stats['mean'][-1]: 3.1f}}}\\"

    left_str += rf"& \ \, \ \pm {stats['rms'][0]:3.1f}"
    rght_str += rf"& \ \, \ \pm {stats['rms'][-1]:3.1f}"
    left_str += r"\end{align}"
    rght_str += r"\end{align}"

    ax.text(1.20, 0.7, left_str, transform=ax.transAxes)
    ax.text(1.85, 0.7, rght_str, transform=ax.transAxes)
    ax.text(-0.60, 0.6, title, transform=ax.transAxes)

    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.tick_params(bottom=False, right=True)
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])
    ax.set_xlim(range_bounds[0] / 1e3 - tick_off, range_bounds[1] / 1e3 + tick_off)
    ax.set_ylim(ylim[0], ylim[1])


inds = (eng.r_a > range_bounds[0]) & (eng.r_a < range_bounds[1])
r_a = eng.r_a[inds]

fig, axes = plt.subplots(4, 1, figsize=(cf.jasa_1clm, 2.25))
ax = axes[0]
lines = (dynamic_eng['bg'] - eng_ri)[:, inds]
sparkline(ax, lines, bg_stats, -0.04, 'BG')

ax.text(-0.62, 1.4, 'Type \hspace{2.2em} dB re RI BG  \hspace{3.1em} 7.5 km  \hspace{1.4em} 47.5 km',
        transform=ax.transAxes)
ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

ax = axes[1]
lines = (dynamic_eng['tilt'] - eng_ri)[:, inds]
sparkline(ax, lines, tilt_stats, -0.04, 'Tilt')

ax = axes[2]
lines = (dynamic_eng['spice'] - eng_ri)[:, inds]
sparkline(ax, lines, spice_stats, -0.04, 'Spice')

ax = axes[3]
lines = (dynamic_eng['total'] - eng_ri)[:, inds]
sparkline(ax, lines, total_stats, -0.04, 'Obs.')

ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
ax.text(0.25, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
ax.tick_params(bottom=True)
ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

fig.savefig(join(savedir, f'shallow_{fc}_blocking.png'), dpi=300)

# Blocking features
dyn = np.array([dynamic_eng[fld] for fld in cf.field_types])
bg_i = eng.blocking_feature(dyn, eng_ri, range_bounds=range_bounds)
block_i = bg_i < 3

fig, axes = plt.subplots(4, 1, figsize=(cf.jasa_1clm, 2.25))
ax = axes[0]
flt_eng = (dynamic_eng['bg'] - eng_ri)[block_i[cf.field_types.index('bg')], :]
flt_bg_stats = eng.field_stats(flt_eng, range_bounds=range_bounds)
sparkline(ax, flt_eng[:, inds], flt_bg_stats, -0.04, 'BG')

ax.text(-0.62, 1.4, 'Type \hspace{2.2em} dB re RI BG  \hspace{3.1em} 7.5 km  \hspace{1.4em} 47.5 km',
        transform=ax.transAxes)
ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

ax = axes[1]
flt_eng = (dynamic_eng['tilt'] - eng_ri)[block_i[cf.field_types.index('tilt')], :]
# remove 3 dB down series
lossy = flt_eng[:, inds][:, 0] < -3
flt_eng = np.delete(flt_eng, lossy, axis=0)
flt_tilt_stats = eng.field_stats(flt_eng, range_bounds=range_bounds)
sparkline(ax, flt_eng[:, inds], flt_tilt_stats, -0.04, 'Tilt')

ax = axes[2]
flt_eng = (dynamic_eng['spice'] - eng_ri)[block_i[cf.field_types.index('spice')], :]
# remove 3 dB down series
lossy = flt_eng[:, inds][:, 0] < -3
flt_eng = np.delete(flt_eng, lossy, axis=0)
flt_spice_stats = eng.field_stats(flt_eng, range_bounds=range_bounds)
sparkline(ax, flt_eng[:, inds], flt_spice_stats, -0.04, 'Spice')

ax = axes[3]
flt_eng = (dynamic_eng['total'] - eng_ri)[block_i[cf.field_types.index('total')], :]
# remove 3 dB down series
lossy = flt_eng[:, inds][:, 0] < -3
flt_eng = np.delete(flt_eng, lossy, axis=0)
flt_total_stats = eng.field_stats(flt_eng, range_bounds=range_bounds)
sparkline(ax, flt_eng[:, inds], flt_total_stats, -0.04, 'Obs.')

ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
ax.text(0.25, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
ax.tick_params(bottom=True)
ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])

fig.savefig(join(savedir, f'shallow_{fc}_no_blocking.png'), dpi=300)

