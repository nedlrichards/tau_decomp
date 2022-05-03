import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src import MLEnergyPE, MLEnergy, Config, list_tl_files, sonic_layer_depth

plt.ion()

fc = 400
cf = Config(fc)

bg_ri_eng = np.load('data/processed/bg_ri_eng.npz', allow_pickle=True)
x_s = bg_ri_eng['x_s']
r_a = bg_ri_eng['r_a']
bg_ri_eng_0 = bg_ri_eng['e_ri_0']
loop_len = bg_ri_eng['loop_len']
bg_ri_eng = bg_ri_eng['e_ri']

tl_list = list_tl_files(fc)
# mode 1 cutoff frequency
m1_co = []
for tl in tl_list:
    tl = np.load(tl)
    c_bg = tl['c_bg']
    bg_prof = np.mean(c_bg, axis=1)
    sld_z, sld_i = sonic_layer_depth(tl['z_a'], bg_prof[:, None], z_max=200)
    sld_i = sld_i[0]

    c_ml = np.array([tl['z_a'][:sld_i], bg_prof[:sld_i]])
    lr = linregress(c_ml)
    dcdz = lr.slope

    m_co = (9/16) * np.sqrt(1500. ** 3 / ((2 * sld_z) ** 3 * dcdz))
    m1_co.append(m_co)

m1_co = np.array(m1_co)

diff_i = (r_a > 5e3) & (r_a < 45e3)
norm_eng = 10 * np.log10(bg_ri_eng * r_a)
norm_eng_0 = 10 * np.log10(bg_ri_eng_0 * r_a)

max_loss = np.min(norm_eng[:, diff_i], axis=1)
max_ll = np.array([np.max(ll) for ll in loop_len])

fig, ax = plt.subplots(figsize=(5,3))
twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(x_s / 1e3, max_loss)
p2, = twin1.plot(x_s / 1e3, m1_co, color='C1')
p3, = twin2.plot(x_s / 1e3, max_ll / 1e3, color='C2')

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
ax.tick_params(axis='x', **tkw)

ax.set_xlabel("Range (km)")
ax.set_ylabel("Min Energy (dB)")
twin1.set_ylabel("Mode 1 cutoff (Hz)")
twin2.set_ylabel("Mode 1 Loop Length (km)")

pos = ax.get_position()
pos.x0 += 0.02
pos.x1 -= 0.14
pos.y0 += 0.04
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig('reports/jasa/figures/bg_duct_metrics.png', dpi=300)

dist_mean = np.mean(norm_eng, axis=0)
dist_rms = np.sqrt(np.var(norm_eng, axis=0))

lin_mean = linregress(r_a[diff_i] / 1e3, y=dist_mean[diff_i])
lin_rms = linregress(r_a[diff_i] / 1e3, y=dist_rms[diff_i])

dist_median = np.median(norm_eng, axis=0)
dist_10 = np.percentile(norm_eng, 10, axis=0, method='median_unbiased')
dist_90 = np.percentile(norm_eng, 90, axis=0, method='median_unbiased')

lin_median = linregress(r_a[diff_i] / 1e3, y=dist_median[diff_i])
lin_10 = linregress(r_a[diff_i] / 1e3, y=dist_10[diff_i])
lin_90 = linregress(r_a[diff_i] / 1e3, y=dist_90[diff_i])

fig, ax = plt.subplots()
ax.plot(r_a[diff_i] / 1e3, norm_eng[:, diff_i].T, '0.7', linewidth=0.5)
ax.plot(r_a[diff_i] / 1e3, norm_eng[:, diff_i].T, '0.7', linewidth=0.5)

ax.plot(r_a[diff_i] / 1e3, dist_mean[diff_i], 'k', linewidth=4)
ax.plot(r_a[diff_i] / 1e3, (dist_mean + dist_rms)[diff_i], 'k', linewidth=3)
ax.plot(r_a[diff_i] / 1e3, (dist_mean - dist_rms)[diff_i], 'k', linewidth=3)

ax.plot(r_a[diff_i] / 1e3, dist_median[diff_i], 'C0', linewidth=4)
ax.plot(r_a[diff_i] / 1e3, dist_10[diff_i], 'C0', linewidth=3)
ax.plot(r_a[diff_i] / 1e3, dist_90[diff_i], 'C0', linewidth=3)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Background excess energy loss (dB)')

pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax.set_position(pos)

fig.savefig('reports/jasa/figures/bg_eng_loss.png', dpi=300)

"""
fig, ax = plt.subplots()
ax.plot(r_a / 1e3, (norm_eng - norm_eng_0).T, 'C0', linewidth=0.5)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Background excess energy loss (dB)')

pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y0 += 0.04
pos.y1 += 0.06
ax.set_position(pos)
"""
