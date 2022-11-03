import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

from src import Config
from src.eng_processing import field_stats, rgs

plt.style.use('elr')
plt.ion()

fc = 400
#fc = 1e3

cf = Config(fc=fc, source_depth='shallow')

load_dir = 'data/processed/'
int_400 = np.load(join(load_dir, 'int_eng_shallow_400.npz'))
int_400_deep = np.load(join(load_dir, 'int_eng_deep_400.npz'))

int_1000 = np.load(join(load_dir, 'int_eng_shallow_1000.npz'))
int_1000_deep = np.load(join(load_dir, 'int_eng_deep_1000.npz'))

r_a = int_400['r_a']

range_bounds = (7.5e3, 47e3)

eng_bg_ml_400 = int_400['ml_ml'][0]
eng_bg_ml_tl_400 = int_400['ml_tl'][0]
eng_bg_tl_ml_400 = int_400_deep['ml_ml'][0]

eng_bg_ml_1000 = int_1000['ml_ml'][0]
eng_bg_ml_tl_1000 = int_1000['ml_tl'][0]
eng_bg_tl_ml_1000 = int_1000_deep['ml_ml'][0]


cmap = plt.cm.magma
i_co = 30
clrs = np.ones(eng_bg_ml_400.shape[0]) * 0.70
clrs[i_co:] = 0.4
clrs = cmap(clrs)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(cf.jasa_2clm, 2.5))

ax = axes[0]
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_400[:i_co, :], axis=0), color=clrs[0], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_400[i_co:, :], axis=0), color=clrs[0])
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_1000[:i_co, :], axis=0), color=clrs[-1], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_1000[i_co:, :], axis=0), color=clrs[-1])

ax.plot([6, 13], [-61, -61], color='k', linestyle='--')
ax.plot([6, 13], [-65, -65], color='k', linestyle='-')
ax.text(15, -62, ' $x_{src}\leq$ 300 km', size=10)
ax.text(15, -66, '       > 300 km', size=10)

ax.text(-10, -26, '(a)', bbox=cf.bbox, clip_on=False)

cax = fig.add_axes([0.15, 0.42, .02, .1])
lcmap = ListedColormap([clrs[0], clrs[-1]])
sm = ScalarMappable(cmap=lcmap)
cb = fig.colorbar(sm, cax=cax, ticks=[0.25, 0.75])
cb.set_ticklabels(['   = 400 Hz', r'$f_c$= 1 kHz'])

x0 = -0.02
dx = 0.05
pos = ax.get_position()
pos.x0 += x0
pos.x1 += x0 + dx
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

ax.set_ylabel('Compensated ML energy (dB)')

ax = axes[1]
ax.plot(r_a / 1e3, np.mean(eng_bg_tl_ml_400[:i_co, :], axis=0), color=clrs[0], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_tl_ml_400[i_co:, :], axis=0), color=clrs[0])
ax.plot(r_a / 1e3, np.mean(eng_bg_tl_ml_1000[:i_co, :], axis=0), color=clrs[-1], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_tl_ml_1000[i_co:, :], axis=0), color=clrs[-1])

ax.text(-3, -26, '(b)', bbox=cf.bbox, clip_on=False)

x0 = 0.0
pos = ax.get_position()
pos.x0 += x0
pos.x1 += x0 + dx
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

ax = axes[2]
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_tl_400[:i_co, :], axis=0), color=clrs[0], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_tl_400[i_co:, :], axis=0), color=clrs[0])
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_tl_1000[:i_co, :], axis=0), color=clrs[-1], linestyle='--')
ax.plot(r_a / 1e3, np.mean(eng_bg_ml_tl_1000[i_co:, :], axis=0), color=clrs[-1])

ax.text(-3, -26, '(c)', bbox=cf.bbox, clip_on=False)

x0 = 0.02
pos = ax.get_position()
pos.x0 += x0
pos.x1 += x0 + dx
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

ax.set_ylim(-70, -25)

fig.supxlabel('Position, $x$ (km)')

fig.savefig('reports/jasa/figures/bg_eng_loss_3_panel.png', dpi=300)
