import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import EngProc, Config

plt.style.use('elr')

plt.ion()

fc = 400
#fc = 1e3
source_depth = "shallow"

load_dir = 'data/processed/'
int_1000 = np.load(join(load_dir, 'int_eng_shallow_1000.npz'))

def blocking_feature(r_a, dynamic_eng, bg_eng, range_bounds=(5e3, 50e3),
                        comp_len=5e3):
    """Compute integrated loss indices blocking features"""
    dr = (r_a[-1] - r_a[0]) / (r_a.size - 1)
    num_int = int(np.ceil(comp_len / dr))

    diff_eng = dynamic_eng - bg_eng
    r_i = (r_a > range_bounds[0]) & (r_a < range_bounds[1])
    diff_eng = diff_eng[:, :, r_i]

    # cumulative loss
    loss = diff_eng[:, :, num_int:] - diff_eng[:, :, :-num_int]
    max_int = np.max(-loss, axis=-1)

    return max_int


def plt_loss(fc):
    int_eng = np.load(join(load_dir, 'int_eng_shallow_' + str(int(fc)) + '.npz'))
    r_a = int_eng['r_a']
    ml_eng = int_eng['ml_ml']
    eng_bg = ml_eng[0]

    range_bounds = (7.5e3, 47.5e3)
    dyn_eng = ml_eng[1:]

    max_int_loss = blocking_feature(r_a, dyn_eng, eng_bg, range_bounds=range_bounds,
                                    comp_len=5e3)
    r_a = int_eng['xs'] / 1e3
    return r_a, max_int_loss


cf = Config()
r_a, mil_400 = plt_loss(400)
r_a, mil_1000 = plt_loss(1e3)
plot_max = 8

fig, axes = plt.subplots(2, 1, figsize=(cf.jasa_2clm,2.5), sharex=True, sharey=True)
axes[0].plot(r_a, mil_400[1:].T, label=cf.field_types[1:])
axes[0].plot(r_a, mil_400[0].T, label=cf.field_types[0])

axes[0].set_prop_cycle(None)
for i, m in enumerate(mil_400[1:]):
    m_i = m > plot_max
    if not np.any(m_i):
        # plot something to cycle colors
        axes[0].plot(-1e3, -1e3)
        continue
    axes[0].plot(r_a[m_i], np.full(m_i.sum(), plot_max - 0.2), '.')

axes[0].plot(r_a, np.full_like(r_a, 3), color='C3', linestyle=':')

axes[1].plot(r_a, mil_1000[1:].T, label=cf.field_types[1:])
axes[1].plot(r_a, mil_1000[0].T, label=cf.field_types[0])

axes[1].set_prop_cycle(None)
for i, m in enumerate(mil_1000[1:]):
    m_i = m > plot_max
    if not np.any(m_i):
        # plot something to cycle colors
        axes[1].plot(-1e3, -1e3)
        continue
    axes[1].plot(r_a[m_i], np.full(m_i.sum(), plot_max - 0.2), '.')

axes[1].plot(r_a, np.full_like(r_a, 3), color='C3', linestyle=':')

fig.supylabel('Maxium loss over 5 km (dB)')
axes[0].set_ylim(-0.5, plot_max)
axes[0].set_xlim(0, 900)
#axes[0].grid()

# Hide the right and top spines
axes[0].spines.right.set_visible(False)
axes[0].spines.top.set_visible(False)

axes[0].set_yticks([0, 3, 6])

axes[1].set_xlabel('Source position (km)')
#axes[1].grid()
#axes[1].legend()

# Hide the right and top spines
axes[1].spines.right.set_visible(False)
axes[1].spines.top.set_visible(False)

axes[1].set_yticks([0, 3, 6])

axes[1].legend(loc=(1.015, 0.6), handlelength=1.0)

x0 = -0.04
dx = -0.02

pos = axes[0].get_position()
pos.x0 += x0
pos.x1 += dx
pos.y0 += 0.07
pos.y1 += 0.07
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 += x0
pos.x1 += dx
pos.y0 += 0.07
pos.y1 += 0.07
axes[1].set_position(pos)

axes[0].text(170, plot_max - 1, '400 Hz', bbox=cf.bbox)
axes[1].text(170, plot_max - 1, '1 kHz', bbox=cf.bbox)

fig.savefig('reports/jasa/figures/integrated_loss.png', dpi=300)
