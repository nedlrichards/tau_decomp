import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

from src import EngProc, Config

from plot_bg import plot_bg

plt.style.use('elr')
plt.ion()

source_depth="shallow"
range_bounds = (7.5e3, 47.5e3)
fc = 400

cf = Config(fc=fc, source_depth=source_depth)
eng = EngProc(cf)

r_a = eng.r_a.copy()
eng_ri = eng.diffraction_bg()
dynamic_eng = eng.dynamic_energy()

bg_stats = eng.field_stats(dynamic_eng['bg'] - eng_ri, range_bounds=range_bounds)
tilt_stats = eng.field_stats(dynamic_eng['tilt'] - eng_ri, range_bounds=range_bounds)
spice_stats = eng.field_stats(dynamic_eng['spice'] - eng_ri, range_bounds=range_bounds)
total_stats = eng.field_stats(dynamic_eng['total'] - eng_ri, range_bounds=range_bounds)

def sparkline(ax, stats):
    ax.plot(stats['r_a'] / 1e3, stats['mean'], linewidth=2, color='k')
    ax.plot(stats['r_a'] / 1e3, stats['mean'] + stats['rms'], linewidth=1, color='k')
    ax.plot(stats['r_a'] / 1e3, stats['mean'] - stats['rms'], linewidth=1, color='k')
    ax.plot(stats['r_a'] / 1e3, stats['10th'], linewidth=1, color='C1', linestyle='--')
    ax.plot(stats['r_a'] / 1e3, stats['90th'], linewidth=1, color='C1', linestyle='--')
    ax.set_ylim(-15, 5)
    ax.axis('off')
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xlim(range_bounds[0] / 1e3, range_bounds[1] / 1e3)

fig, axes = plt.subplots(4, 1, figsize=(cf.jasa_1clm, 2))
ax = axes[0]
ax.plot(r_a / 1e3, (dynamic_eng['bg'] - eng_ri).T, linewidth=0.5, color='0.8')
sparkline(ax, bg_stats)

ax = axes[1]
ax.plot(r_a / 1e3, (dynamic_eng['tilt'] - eng_ri).T, linewidth=0.5, color='0.8')
sparkline(ax, tilt_stats)


ax = axes[2]
ax.plot(r_a / 1e3, (dynamic_eng['spice'] - eng_ri).T, linewidth=0.5, color='0.8')
sparkline(ax, spice_stats)


ax = axes[3]
ax.plot(r_a / 1e3, (dynamic_eng['total'] - eng_ri).T, linewidth=0.5, color='0.8')
sparkline(ax, total_stats)


