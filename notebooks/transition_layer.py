import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os

from src import MLEnergy, list_tl_files, Config, sonic_layer_depth

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)
test_field = 'total'

r_bound = (7.5e3, 47.5e3)

tl_file = tl_files[15]
ml_eng = MLEnergy(tl_file)

# start with range independent ssp
rd_modes = ml_eng.field_modes['bg']
ssp = rd_modes.bg_prof
z_a = rd_modes.z_a
dz = rd_modes.dz
sld_z, sld_i = sonic_layer_depth(z_a, ssp[:, None], z_max=200)
sld_i = int(sld_i)

sld_z = sld_z[0]
c_sld = ssp[sld_i]
kx = 2 * pi * fc / c_sld

# trace beam through transition layer
z_min = sld_z
z_max = 150.
z_i = (z_a > sld_z) & (z_a < z_max)
z_transition = z_a[z_i]
ssp_transition = ssp[z_i]
ky = np.sqrt((2 * pi * fc / ssp_transition) ** 2 - kx ** 2)
dx = dz * (kx / ky)
total_distance = np.sum(dx)

"""Energy from mixed layer computed by PE"""
z_pe = ml_eng.z_a
z_ml_pe = z_pe < z_min
p_ml = ml_eng.tl_data['p_' + test_field][:, z_ml_pe]
en_ml = np.sum(np.abs(p_ml ** 2), axis=1) * ml_eng.dz

z_tl_pe = (z_pe > z_min) & (z_pe < z_max)
p_tl = ml_eng.tl_data['p_' + test_field][:, z_tl_pe]
en_tl = np.sum(np.abs(p_tl ** 2), axis=1) * ml_eng.dz

r_match = 7.5e3

r_a = ml_eng.r_a
dx = (r_a[-1] - r_a[0]) / (r_a.size - 1)
r_match_i = np.argmin(np.abs(r_match - r_a))

num_points = 60
decimation = int(r_a.size / num_points)
en_model = np.zeros(en_ml.size - 1)

en_model = en_ml[:-1] * r_a[:-1]
en_model /= r_a[1:]

n_r_tl = int(total_distance / dx)
tl_in = en_model - en_ml[1:]

acc_model = np.cumsum(tl_in)
acc_model[n_r_tl:] = acc_model[n_r_tl:] - acc_model[:-n_r_tl]
acc_model = acc_model[n_r_tl - 1:]

test = [np.sum(tl_in[i: i + n_r_tl]) for i in range(tl_in[:-n_r_tl].size)]

test_1 = np.zeros(r_a[n_r_tl:-1].size)
test_2 = np.zeros(r_a[n_r_tl:-1].size)
for i in range(tl_in[:-n_r_tl].size):
    max_i = min(i + n_r_tl, test_2.size - 1)
    test_1[i: max_i] += tl_in[i] * r_a[i] / r_a[i: max_i]
    #if i % n_r_tl == 0:
    test_2[i: max_i] += tl_in[i] * (r_a[i] / r_a[i: max_i]) ** 2

#test_2 = [np.sum(tl_in[i: i + n_r_tl] * r_a[i] / r_a[i: i + n_r_tl]) for i in range(tl_in[:-n_r_tl].size)]


fig, ax = plt.subplots()
#ax.plot(r_a, 10 * np.log10(en_ml * r_a))
#ax.plot(r_a[1:], 10 * np.log10(en_model * r_a[1:]))
ax.plot(r_a, 10 * np.log10(en_tl * r_a))
#ax.plot(r_a[n_r_tl:], 10 * np.log10(acc_model * r_a[n_r_tl:]))
ax.plot(r_a[n_r_tl:-1], 10 * np.log10(test * r_a[n_r_tl:-1]))
#ax.plot(r_a[n_r_tl:-1], 10 * np.log10(test_2 * r_a[n_r_tl:-1]))
ax.plot(r_a[:-n_r_tl - 1], 10 * np.log10(test_1 * r_a[n_r_tl:-1]))
ax.plot(r_a[:-n_r_tl-1], 10 * np.log10(test_2 * r_a[n_r_tl:-1]))
#ax.plot(r_a[:-n_r_tl], 10 * np.log10(acc_model * r_a[:-n_r_tl]))
#ax.plot(r_a[n_r_tl:-1], 10 * np.log10(test * r_a[n_r_tl:-1]))
#ax.plot(r_a[1:], 10 * np.log10((en_model - en_ml[1:]) * r_a[1:]))

