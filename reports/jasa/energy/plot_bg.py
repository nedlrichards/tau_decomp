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

def plot_bg(eng_obj, eng_bg, range_bounds):
    """method to plot energy realizations and statistics"""

    cmap = plt.cm.cividis
    i_co = 30
    clrs = np.ones(eng_bg.shape[0]) * 0.85
    clrs[i_co:] = 0.5
    alpha = np.ones(eng_bg.shape[0])
    alpha[i_co:] = 0.4
    clrs = cmap(clrs)

    fig, ax = plt.subplots(figsize=(eng_obj.cf.jasa_1clm, 2.5))
    for i, (e, c, a) in enumerate(zip(eng_bg, clrs, alpha)):
        ax.plot(eng_obj.r_a / 1e3, e, color=c, alpha=a, linewidth=0.5)

    bg_stats = eng_obj.field_stats(eng_bg,
                                   range_bounds=range_bounds)

    r_a_plt , mean_rgs = eng_obj.rgs('mean', bg_stats, range_bounds=range_bounds,
                                scale_r=True)

    ax.plot(r_a_plt, mean_rgs, 'k', linewidth=2)

    _ , rms_rgs = eng_obj.rgs('rms', bg_stats, range_bounds=range_bounds,
                            scale_r=True)
    ax.plot(r_a_plt, rms_rgs, 'k', linewidth=2)
    ax.plot(r_a_plt, 2 * mean_rgs - rms_rgs, 'k', linewidth=1)

    ax.plot(*eng_obj.rgs('10th', bg_stats, range_bounds=range_bounds, scale_r=True),
            'C1', linewidth=1.5, linestyle='--')
    ax.plot(*eng_obj.rgs('90th', bg_stats, range_bounds=range_bounds, scale_r=True),
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
    return fig

