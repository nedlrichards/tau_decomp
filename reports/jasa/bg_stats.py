import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src import MLEnergyPE, Config, list_tl_files

plt.style.use('elr')
plt.ion()

fc = 400
#fc = 1e3
source_depth="shallow"
#source_depth="deep"
cf = Config(fc=fc, source_depth=source_depth)

eng_bg = []
for tl in list_tl_files(fc, source_depth=source_depth):
    ml_pe = MLEnergyPE(tl)
    eng_bg.append(10 * np.log10(ml_pe.ml_energy('bg') * ml_pe.r_a))
r_a = ml_pe.r_a
eng_bg = np.array(eng_bg)

# estimate stats away from source and CZs
diff_i = (r_a > 5e3) & (r_a < 50e3)

bg_mean = np.mean(eng_bg[:, diff_i], axis=0)
bg_var = np.var(eng_bg[:, diff_i], axis=0)
bg_10 = np.percentile(eng_bg[:, diff_i], 10, axis=0, method='median_unbiased')
bg_90 = np.percentile(eng_bg[:, diff_i], 90, axis=0, method='median_unbiased')

bg_mean_rgs = linregress(r_a[diff_i], y=bg_mean)
bg_rms_rgs = linregress(r_a[diff_i], y=bg_mean + np.sqrt(bg_var))
bg_10_rgs = linregress(r_a[diff_i], y=bg_10)
bg_90_rgs = linregress(r_a[diff_i], y=bg_90)

def rgs(lin_rgs):
    """compute linear regression line from object"""
    return lin_rgs.intercept + r_a[diff_i] * lin_rgs.slope

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.5))
ax.plot(r_a / 1e3, eng_bg.T, '0.7', linewidth=0.5)

ax.plot(r_a[diff_i] / 1e3, rgs(bg_mean_rgs), 'k', linewidth=2)
ax.plot(r_a[diff_i] / 1e3, rgs(bg_rms_rgs), 'k', linewidth=1)
ax.plot(r_a[diff_i] / 1e3, 2 * rgs(bg_mean_rgs) - rgs(bg_rms_rgs), 'k', linewidth=1)

ax.plot(r_a[diff_i] / 1e3, rgs(bg_10_rgs), 'C1', linewidth=1, linestyle='--')
ax.plot(r_a[diff_i] / 1e3, rgs(bg_90_rgs), 'C1', linewidth=1, linestyle='--')

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

print(f'Mean slope: {bg_mean_rgs.slope * 1e3:.3f}' +
      f' $\pm$ {(bg_rms_rgs.slope - bg_mean_rgs.slope) * 1e3:.3f} (dB / km)')
print(f'90th percentile slope: {bg_90_rgs.slope * 1e3:.3f}')
print(f'10th percentile slope: {bg_10_rgs.slope * 1e3:.3f}')

