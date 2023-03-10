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
source_depth="shallow"

cf = Config(fc=fc, source_depth=source_depth)

load_dir = 'data/processed/'

int_400 = np.load(join(load_dir, 'int_eng_' + source_depth + '_400.npz'))

r_a = int_400['r_a']

range_bounds = (7.5e3, 47e3)
eng_bg = int_400['ml_ml'][0]

bg_stats = field_stats(r_a, eng_bg, range_bounds=range_bounds)

r_a_plt , mean_rgs = rgs(r_a, 'mean', bg_stats, range_bounds=range_bounds, scale_r=True)
_ , rms_rgs = rgs(r_a, 'rms', bg_stats, range_bounds=range_bounds, scale_r=True)


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

ax.plot(r_a_plt, mean_rgs, 'k', linewidth=2)


ax.plot(r_a_plt, mean_rgs + rms_rgs, 'C1', linewidth=1.5, linestyle='--')
ax.plot(r_a_plt, mean_rgs - rms_rgs, 'C1', linewidth=1.5, linestyle='--')


#ax.plot(*eng.rgs('10th', bg_stats, range_bounds=range_bounds, scale_r=True),
        #'C1', linewidth=1.5, linestyle='--')
#ax.plot(*eng.rgs('90th', bg_stats, range_bounds=range_bounds, scale_r=True),
        #'C1', linewidth=1.5, linestyle='--')

cax = fig.add_axes([0.25, 0.3, .02, .1])
lcmap = ListedColormap([clrs[0], clrs[-1]])
sm = ScalarMappable(cmap=lcmap)
cb = fig.colorbar(sm, cax=cax, ticks=[0.25, 0.75])
cb.set_ticklabels([r'$x_{src}\leq$300 km', '$x_{src}$ >300km'])
ax.set_ylim(-40, -25)

ax.set_xlabel('Position, $x$ (km)')
ax.set_ylabel('MLAD energy (dB)')

pos = ax.get_position()
pos.x0 += 0.10
pos.x1 += 0.05
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/bg_eng_loss.png', dpi=300)
