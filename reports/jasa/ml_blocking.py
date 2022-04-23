import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt

from src import MLEnergyPE, MLEnergy, Config, list_tl_files

plt.ion()

fc = 400
cf = Config(fc)

bg_ri_eng = np.load('data/processed/bg_ri_eng.npz')
diff_eng = bg_ri_eng['e_ri']

tl_list = list_tl_files(fc)

pe_ml_engs = []

x_s = []
all_eng = []
for r in tl_list:
    e = MLEnergyPE(cf.fc, r)
    x_s.append(e.xs)
    o_r = np.array([e.ml_energy(ft) for ft in cf.field_types])
    all_eng.append(o_r[:, None, :])

r_a = e.r_a
x_s = np.array(x_s)

all_eng = np.concatenate(all_eng, axis=1)

norm_eng = np.log10(all_eng) - np.log10(diff_eng)
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

fig, ax = plt.subplots(figsize=(cf.jasa_1clm,3))
ax.plot(x_s / 1e3, max_int[1:, :].T, marker='.')
ax.set_xlabel('Starting position (km)')
ax.set_ylabel('Maxium loss over 5 km (dB)')
ax.set_ylim(-0.5, 18)
ax.set_xlim(0, 900)
ax.grid()
ax.legend(['tilt', 'spice', 'observed'])

pos = ax.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 += 0.04
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig('reports/jasa/figures/integrated_loss.png', dpi=300)

# threshold loss features
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(cf.jasa_1clm,3))
m_i = max_int[1, :] > 3
ax[0].plot(r_a / 1e3, norm_eng[1, m_i, :].T, linewidth=.5, color='0.2')
ax[0].plot(r_a / 1e3, norm_eng[1, ~m_i, :].T, linewidth=.5, color='C0')

m_i = max_int[2, :] > 3
ax[1].plot(r_a / 1e3, norm_eng[2, m_i, :].T, linewidth=.5, color='0.2')
ax[1].plot(r_a / 1e3, norm_eng[2, ~m_i, :].T, linewidth=.5, color='C0')

m_i = max_int[3, :] > 3
ax[2].plot(r_a / 1e3, norm_eng[3, m_i, :].T, linewidth=.5, color='0.2')
ax[2].plot(r_a / 1e3, norm_eng[3, ~m_i, :].T, linewidth=.5, color='C0')

ax[0].set_xlim(5, 45)
ax[1].set_ylim(-15, 5)

ax[0].text(5, 3, '(a)', bbox=cf.bbox)
ax[1].text(5, 3, '(b)', bbox=cf.bbox)
ax[2].text(5, 3, '(c)', bbox=cf.bbox)

fig.supylabel('Compensated ML energy')
ax[2].set_xlabel('Range (km)')

pos = ax[0].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax[2].set_position(pos)

fig.savefig('reports/jasa/figures/ml_energy.png', dpi=300)
