import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergyPE, MLEnergy, Config, list_tl_files
plt.style.use('elr')

plt.ion()

fc = 400
#fc = 1e3
source_depth = "shallow"
cf = Config(source_depth=source_depth, fc=fc)

bg_ri_eng = np.load("data/processed/bg_ri_eng_" + source_depth + ".npz")
bg_ri_eng = bg_ri_eng[f'e_ri_{int(fc)}']

tl_list = list_tl_files(fc, source_depth=source_depth)

pe_ml_engs = []

x_s = []
all_eng = []
for r in tl_list:
    e = MLEnergyPE(r)
    x_s.append(e.xs)
    o_r = np.array([e.ml_energy(ft) for ft in cf.field_types])
    all_eng.append(o_r[:, None, :])

r_a = e.r_a
x_s = np.array(x_s)

all_eng = np.concatenate(all_eng, axis=1)

norm_eng = np.log10(all_eng) - np.log10(bg_ri_eng)
norm_eng *= 10

dr = (r_a[-1] - r_a[0]) / (r_a.size - 1)
diff_i = (r_a > 5e3) & (r_a < 45e3)

# strange transpose arrises from indexing
diff_eng = np.diff(norm_eng[:, :, diff_i], axis=-1) / dr

win_len = 50
move_sum = np.cumsum(diff_eng, dtype=float, axis=-1)
move_sum[:, :, win_len:] = move_sum[:, :, win_len:] - move_sum[:, :, :-win_len]
move_sum = move_sum[:, :, win_len - 1:] * dr

max_int = np.max(-move_sum, axis=-1)

plt_max_int = max_int[1:, :].T
plt_max_int = plt_max_int[:, ::-1]

fig, ax = plt.subplots(figsize=(cf.jasa_1clm,2.5))
#ax.plot(x_s / 1e3, max_int[1:, :].T, marker='.')
ax.plot(x_s / 1e3, plt_max_int)
ax.set_xlabel('Starting position (km)')
ax.set_ylabel('Maxium loss over 5 km (dB)')
ax.set_ylim(-0.5, 15)
ax.set_xlim(0, 900)
ax.grid()
#ax.legend(['tilt', 'spice', 'observed'])

pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/integrated_loss.png', dpi=300)
