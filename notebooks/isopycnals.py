import numpy as np
import matplotlib.pyplot as plt

from src import Config
from src import Field
from src import lvl_profiles

plt.style.use('elr')

plt.ion()
cf = Config()
field = Field()

d_iso = 0.01  # fine isopycnals spacing
sig_start = np.ceil(field.xy_sig.min() / d_iso) * d_iso
sig_end = np.floor(field.xy_sig.max() / d_iso) * d_iso
sig_lvls = np.arange(sig_start, sig_end, d_iso)[:-3]

lvls = lvl_profiles(field.z_a, field.xy_sig, field.xy_gamma, sig_lvls)

top_i = 49
"""
fig, ax = plt.subplots()
ax.plot(field.x_a / 1e3, lvls[0, top_i : top_i + 8, :].T)
#ax.set_ylim(150, 0)
ax.set_ylim(120, 60)
ax.set_xlim(0, 100)

"""
# stable contours
top_cntr = [[14, 923], [19, 905], [20, 757], [21, 710], [23, 618],
            [21, 710], [23, 618], [26, 612], [28, 601], [30, 564],
            [31, 526], [33, 514], [36, 460], [38, 418], [40, 402],
            [42, 268], [44, 205], [48, 129], [50, 55], [53, 0]]

fig, ax = plt.subplots()
ax.plot(field.x_a / 1e3, lvls[0, :, :].T)

last_i = field.x_a.size
for tc in top_cntr:
    p_i = np.arange(tc[1], last_i)
    ax.plot(field.x_a[p_i] / 1e3, lvls[0, tc[0], p_i].T, linewidth=2, color='k')
    last_i = tc[1]


ax.set_ylim(130, 50)
ax.set_xlim(800, 1000)


