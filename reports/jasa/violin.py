import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

from scipy.stats import linregress

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

bg_i = cf.field_types.index('bg')
tilt_i = cf.field_types.index('tilt')
spice_i = cf.field_types.index('spice')
total_i = cf.field_types.index('total')
type_i = [bg_i, tilt_i, spice_i, total_i]

lines_400 = shallow_int_400['ml_ml']
demean_400 = lines_400[1:] - lines_400[0]
stats_400 = [field_stats(r_a, v, range_bounds=range_bounds) for v in demean_400]
stats_400_block = [field_stats(r_a, v[bi], range_bounds=range_bounds) for v, bi in zip(demean_400, block_i)]

lines_400 = shallow_int_400['ml_proj']
demean_400 = lines_400[1:] - lines_400[0]
stats_400_modes = [field_stats(r_a, v, range_bounds=range_bounds) for v in demean_400]
stats_400_modes_block = [field_stats(r_a, v[bi], range_bounds=range_bounds) for v, bi in zip(demean_400, block_i)]

x0 = range_bounds[0]
x1 = range_bounds[1]

def plot_stats(stat, plot_i, ax0, ax1):
    rgs = stat["mean_rgs"]
    m = rgs.slope * range_bounds + rgs.intercept
    rgs = stat["rms_rgs"]
    s = rgs.slope * range_bounds + rgs.intercept

    ax0.plot([plot_i, plot_i], [m[0] - s[0], m[0] + s[0]], 'k')
    ax0.plot([plot_i], [m[0]], 'ok')

    ax1.plot([plot_i, plot_i], [m[1] - s[1], m[1] + s[1]], 'k')
    ax1.plot([plot_i], [m[1]], 'ok')


fig, axes = plt.subplots(1, 2, sharey=True)

for s_i in range(4):
    i = s_i * 4
    t_i = type_i[s_i]
    plot_stats(stats_400[t_i], i, axes[0], axes[1])
    plot_stats(stats_400_modes[t_i], i + 1, axes[0], axes[1])
    plot_stats(stats_400_block[t_i], i + 2, axes[0], axes[1])
    plot_stats(stats_400_modes_block[t_i], i + 3, axes[0], axes[1])

