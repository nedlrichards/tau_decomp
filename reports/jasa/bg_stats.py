import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress

from src import EngProc, Config

plt.style.use('elr')
plt.ion()

fc = 400
#fc = 1e3
source_depth="shallow"
#source_depth="deep"

cf = Config(fc=fc, source_depth=source_depth)
eng = EngProc(cf)

r_a = eng.r_a
eng_bg = eng.bg_eng

range_bounds = (5e3, 50e3)
bg_stats = eng.field_stats(eng_bg, range_bounds=range_bounds)

cmap = plt.cm.cividis
i_co = 30
clrs = np.ones(eng_bg.shape[0]) * 0.85
clrs[i_co:] = 0.5
alpha = np.ones(eng_bg.shape[0])
alpha[i_co:] = 0.4
clrs = cmap(clrs)

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.5))

for i, (e, c, a) in enumerate(zip(eng_bg, clrs, alpha)):
    ax.plot(r_a / 1e3, e, color=c, alpha=a, linewidth=0.5)

r_a_plt , mean_rgs = eng.rgs('mean', bg_stats, range_bounds=range_bounds,
                             scale_r=True)
ax.plot(r_a_plt, mean_rgs, 'k', linewidth=2)

_ , rms_rgs = eng.rgs('rms', bg_stats, range_bounds=range_bounds,
                         scale_r=True)
ax.plot(r_a_plt, rms_rgs, 'k', linewidth=2)
ax.plot(r_a_plt, 2 * mean_rgs - rms_rgs, 'k', linewidth=1)

ax.plot(*eng.rgs('10th', bg_stats, range_bounds=range_bounds, scale_r=True),
        'C1', linewidth=1.5, linestyle='--')
ax.plot(*eng.rgs('90th', bg_stats, range_bounds=range_bounds, scale_r=True),
        'C1', linewidth=1.5, linestyle='--')

cax = fig.add_axes([0.25, 0.3, .02, .1])
lcmap = ListedColormap([clrs[0], clrs[-1]])
sm = ScalarMappable(cmap=lcmap)
cb = fig.colorbar(sm, cax=cax, ticks=[0.25, 0.75])
cb.set_ticklabels([r'$x_{src}\leq$300 km', '$x_{src}$ >300km'])
ax.set_ylim(-18, -10)

ax.set_xlabel('Position, $x$ (km)')
ax.set_ylabel('Compensated ML energy (dB)')

pos = ax.get_position()
pos.x0 += 0.06
pos.x1 += 0.05
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/bg_eng_loss.png', dpi=300)

print(f"Mean slope: {bg_stats['mean_rgs'].slope * 1e3:.3f}" +
      f" $\pm$ {(bg_stats['rms_rgs'].slope - bg_stats['mean_rgs'].slope) * 1e3:.3f} (dB / km)")

print(f"90th percentile slope: {bg_stats['90th_rgs'].slope * 1e3:.3f}")
print(f"10th percentile slope: {bg_stats['10th_rgs'].slope * 1e3:.3f}")

