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

range_bounds = (7.5e3, 40.0e3)
fc = 1000
savedir = 'reports/jasa/figures'
cf = Config()

eng_1 = EngProc(Config(fc=1000, source_depth='shallow'), fields=['bg'])
e_ri_1 = eng_1.diffraction_bg()

eng_400 = EngProc(Config(fc=400, source_depth='shallow'), fields=['bg'])
e_ri_400 = eng_400.diffraction_bg()

eng_1 = EngProc(Config(fc=1000, source_depth='deep'), fields=['bg'])
e_ri_1_ref = eng_1.diffraction_bg()
dyn_1 = eng_1.dynamic_energy()

eng_400 = EngProc(Config(fc=400, source_depth='deep'), fields=['bg'])
e_ri_400_ref = eng_400.diffraction_bg()
dyn_400 = eng_400.dynamic_energy()

lines_400 = {'bg':dyn_400['bg'] - e_ri_400_ref,
             'tilt':dyn_400['tilt'] - e_ri_400_ref,
             'spice':dyn_400['spice'] - e_ri_400_ref,
             'total':dyn_400['total'] - e_ri_400_ref}
lines_1 = {'bg':dyn_1['bg'] - e_ri_1_ref,
           'tilt':dyn_1['tilt'] - e_ri_1_ref,
           'spice':dyn_1['spice'] - e_ri_1_ref,
           'total':dyn_1['total'] - e_ri_1_ref}

stats_400 = {i:eng_400.field_stats(v, range_bounds=range_bounds) for i, v in lines_400.items()}
stats_1 = {i:eng_1.field_stats(v, range_bounds=range_bounds) for i, v in lines_1.items()}

tick_off = 1

def sparkline(ax, lines, stats, dy, title, ylim=(-15, 15), dx=0):
    inds = (stats['r_a'] > range_bounds[0]) & (stats['r_a'] < range_bounds[1])
    r_a = stats['r_a'][inds]

    ax.plot(r_a / 1e3, lines.T, linewidth=0.5, color='0.7')

    # mark lines crossing past ylim
    cross_i = np.argmax(lines < ylim[0], axis=-1)
    cross_r = r_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[0]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')

    cross_i = np.argmax(lines < ylim[-1], axis=-1)
    cross_r = r_a[cross_i[cross_i > 0]]
    ax.plot(cross_r / 1e3, np.full_like(cross_r, ylim[-1]), color='0.7',
            marker='.', linestyle='None', markeredgecolor='0.5')


    ax.plot(r_a / 1e3, stats['mean'][inds], linewidth=1.5, color='k')
    ax.plot(r_a / 1e3, (stats['mean'] + stats['rms'])[inds], linewidth=0.75, linestyle='--', color='k')
    ax.plot(r_a / 1e3, (stats['mean'] - stats['rms'])[inds], linewidth=0.75, linestyle='--', color='k')
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

    ax.text(1.10, 0.7, left_str, transform=ax.transAxes)
    ax.text(1.60, 0.7, rght_str, transform=ax.transAxes)
    ax.text(-0.48, 0.6, title, transform=ax.transAxes)

    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.tick_params(bottom=False, right=True)
    ax.set(yticks=[-15, 15])
    ax.set(yticklabels=[])
    ax.set(xticklabels=[])
    ax.set_xlim(range_bounds[0] / 1e3 - tick_off, range_bounds[1] / 1e3 + tick_off)
    ax.set_ylim(ylim[0], ylim[1])

inds = (eng_400.r_a > range_bounds[0]) & (eng_400.r_a < range_bounds[1])
r_a = eng_400.r_a[inds]

fig, axes = plt.subplots(4, 2, figsize=(cf.jasa_2clm, 2.25))
ax = axes[0, 0]
sparkline(ax, lines_400['bg'][:, inds], stats_400['bg'], -0.04, 'BG')

ax.text(-0.48, 1.4, 'Type \hspace{3.4em} 400 Hz, dB re RI BG  \hspace{1.7em} 7.5 km  \hspace{2.0em} 47.5 km',
        transform=ax.transAxes)
ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize':8})

ax = axes[1, 0]
sparkline(ax, lines_400['tilt'][:, inds], stats_400['tilt'], -0.04, 'Tilt')
#ax.plot(r_a / 1e3, np.mean(e_ri_400_ref[:, inds] - e_ri_400[:, inds], axis=0), color='C1', linewidth=0.5)

ax = axes[2, 0]
sparkline(ax, lines_400['spice'][:, inds], stats_400['spice'], -0.04, 'Spice')

ax = axes[3, 0]
sparkline(ax, lines_400['total'][:, inds], stats_400['total'], -0.04, 'Obs.')
ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
ax.tick_params(bottom=True)

ax = axes[0, 1]
sparkline(ax, lines_1['bg'][:, inds], stats_1['bg'], -0.04, '', dx=0.04)

ax.text(0.10, 1.4, '1 kHz, dB re RI BG  \hspace{2.0em} 7.5 km  \hspace{2.0em} 47.5 km',
        transform=ax.transAxes)

ax = axes[1, 1]
sparkline(ax, lines_1['tilt'][:, inds], stats_1['tilt'], -0.04, '', dx=0.04)

ax = axes[2, 1]
sparkline(ax, lines_1['spice'][:, inds], stats_1['spice'], -0.04, '', dx=0.04)

ax = axes[3, 1]
sparkline(ax, lines_1['total'][:, inds], stats_1['total'], -0.04, '', dx=0.04)

ax.set_xticks([range_bounds[0] / 1e3, range_bounds[1] / 1e3])
ax.set_xticklabels(ax.get_xticks(), fontdict={'fontsize':8})
ax.text(0.30, -0.65, 'Range (km)', transform=ax.transAxes, fontsize=8)
ax.tick_params(bottom=True)

fig.savefig(join(savedir, f'deep_source_energy.png'), dpi=300)

