import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src import MLEnergyPE, MLEnergy, Config, list_tl_files, sonic_layer_depth

plt.style.use('elr')
plt.ion()

fc = 400
#fc = 1e3
source_depth="shallow"
#source_depth="deep"
cf = Config(fc=fc, source_depth=source_depth)

fields = {'bg':[], 'tilt':[], 'spice':[], 'total':[]}

ftype = cf.field_types.copy()
ftype.remove('bg')

eng_bg = []
for tl in list_tl_files(fc, source_depth='shallow'):
    ml_pe = MLEnergyPE(tl)
    eng_bg.append(10 * np.log10(ml_pe.ml_energy('bg') * ml_pe.r_a))

eng_bg = np.array(eng_bg)
r_a = ml_pe.r_a
diff_i = (r_a > 5e3) & (r_a < 45e3)

for tl in list_tl_files(fc, source_depth=source_depth):
    ml_pe = MLEnergyPE(tl)
    for fld in ftype:
        fields[fld].append(10 * np.log10(ml_pe.ml_energy(fld) * ml_pe.r_a))

field_diff = []
for fld in ftype:
    field_diff.append(np.array(fields[fld]) - eng_bg)
field_diff = np.array(field_diff)

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

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)

if source_depth == 'shallow':
    ax[2].set_ylim(-10, 5)
    txt_y = 5
else:
    ax[2].set_ylim(-25, -3)
    txt_y = -5

for i, fld in enumerate(ftype):
    ax[i].plot(r_a / 1e3, field_diff[i].T, '0.7', linewidth=0.5)
    ax[i].text(1, txt_y, fld, bbox=cf.bbox)

ax[2].set_xlim(0, 60)
ax[2].set_xlabel('Position, $x$ (km)')

fig.supylabel('ML energy (dB re background)')

pos = ax[0].get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax[2].set_position(pos)

