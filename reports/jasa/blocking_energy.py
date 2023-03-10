import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.signal import find_peaks, hilbert
from scipy.interpolate import interp1d
from scipy.stats import linregress

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

import pickle

plt.ion()
plt.style.use('elr')

save_dir = 'data/processed/'
range_bounds = (7.5e3, 47e3)
total_rb = (7.5e3, 47e3)
fc = 400
#fc = 1e3
cf = Config(fc=fc, source_depth='shallow')

tl_ind = 23
field_type = 'total'

eng_ind = cf.field_types.index(field_type) + 1

load_dir = 'data/processed/'
shallow_int_400 = np.load(join(load_dir, 'int_eng_shallow_' + str(int(fc)) + '.npz'))
deep_int_400 = np.load(join(load_dir, 'int_eng_deep_' + str(int(fc)) + '.npz'))

r_a = shallow_int_400['r_a']
plot_r_a = shallow_int_400['r_a'] + shallow_int_400['xs'][tl_ind]
rb_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])

ml_400 = shallow_int_400['ml_ml']
proj_400 = shallow_int_400['ml_proj']
tl_400 = shallow_int_400['ml_tl']

ml_bg = ml_400[0]
tl_bg = tl_400[0]
deep_bg = deep_int_400['ml_ml'][0]

ml_eng = ml_400[eng_ind]
tl_eng = tl_400[eng_ind]
proj_eng = proj_400[eng_ind]
deep_eng = deep_int_400['ml_ml'][eng_ind]

demean = []
tl_bg_fit = []
for bg, eng in zip(tl_bg, tl_eng):
    fit = np.polynomial.polynomial.polyfit(r_a[rb_i], bg[rb_i], 1)
    tl_bg_fit.append(np.polynomial.polynomial.polyval(r_a, fit))
    demean.append(eng - tl_bg_fit[-1])
demean = np.array(demean)
tl_bg_fit = np.array(tl_bg_fit)

rb_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])
fit = np.polynomial.polynomial.polyfit(r_a[rb_i], tl_eng[tl_ind][rb_i], 1)
tl_fit = np.polynomial.polynomial.polyval(r_a, fit)

exp_i = (r_a >= range_bounds[1]) & (r_a <= total_rb[1])
ml_rb = (r_a >= total_rb[0]) & (r_a <= total_rb[1])
fit = np.polynomial.polynomial.polyfit(r_a[ml_rb], ml_eng[tl_ind][ml_rb], 1)
ml_fit = np.polynomial.polynomial.polyval(r_a, fit)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(cf.jasa_1clm, 3))
ax = axes[0]
l0 = ax.plot(plot_r_a / 1e3, ml_bg[tl_ind], color='k', label='BG')
l1 = ax.plot(plot_r_a / 1e3, ml_eng[tl_ind], color='C1', label='CMLE')
l2 = ax.plot(plot_r_a / 1e3, proj_eng[tl_ind] - 10 * np.log10(92), color='C1', label='MLM1', linewidth=1, linestyle=':')

ax.plot(plot_r_a / 1e3, tl_bg[tl_ind], color='k')
l3 = ax.plot(plot_r_a / 1e3, tl_eng[tl_ind], color='C2', label='CTRLE')

ax.set_ylim(-70, -20)

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.06
pos.y0 += 0.07
pos.y1 += 0.07
ax.set_position(pos)
ax.grid()

ax.text(231, -22, '(a)', bbox=cf.bbox)

ax = axes[1]
ax.plot(plot_r_a / 1e3, deep_bg[tl_ind], color='k', label='BG')
l4 = ax.plot(plot_r_a / 1e3, deep_eng[tl_ind], color='r', label='CMLE')

#ax.legend(handles=l0 + l1 + l2 + l3 + l4, fontsize=8, framealpha=1.0,
          #loc='lower left', bbox_to_anchor=(0.65, 0.50), handlelength=1)

l = ax.legend(handles=l0 + l1 + l2 + l3, fontsize=8, framealpha=1.0,
              loc='lower left', bbox_to_anchor=(0.78, 0.75), handlelength=1)
ax.add_artist(l)


ax.legend(handles=l0 + l4, fontsize=8, framealpha=1.0,
          loc='lower left', bbox_to_anchor=(0.78, -0.10), handlelength=1)

ax.set_ylim(-70, -20)
ax.set_xlim(plot_r_a[0]/1e3 - 1, plot_r_a[0]/1e3 + 55)

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.06
pos.y0 += 0.07
pos.y1 += 0.07
ax.set_position(pos)
ax.grid()

ax.text(231, -22, '(b)', bbox=cf.bbox)

fig.supxlabel('Position (x)')
fig.supylabel('Verically averaged energy (dB)')

#ax.legend([l0, l1, l2], ['RI BG', 'MLAD', 'TL'])
#ax.legend([l0, l1, l2])

savedir = 'reports/jasa/figures'
fig.savefig(join(savedir, 'blocking_energy.png'), dpi=300)
