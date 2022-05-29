import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import EngProc

plt.style.use('elr')

plt.ion()

fc = 400
#fc = 1e3

source_depth = "shallow"
#source_depth = "deep"

eng = EngProc(fc=fc, source_depth=source_depth)

cf = eng.cf
r_a = eng.r_a
eng_bg = eng.bg_eng

range_bounds = (5e3, 50e3)
max_int_loss = eng.blocking_feature(range_bounds=range_bounds,
                                    comp_len=5e3)

fig, ax = plt.subplots(figsize=(cf.jasa_1clm,2.5))
#ax.plot(x_s / 1e3, max_int.T)
ax.plot(eng.xs / 1e3, max_int_loss.T)
ax.set_xlabel('Starting position (km)')
ax.set_ylabel('Maxium loss over 5 km (dB)')
ax.set_ylim(-0.5, 15)
ax.set_xlim(0, 900)
ax.grid()
ax.legend(eng.dy_fields)
ax.set_yticks([0, 3, 6, 9, 12])
pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/integrated_loss.png', dpi=300)
