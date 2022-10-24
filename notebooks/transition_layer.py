import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os

from src import MLEnergy, list_tl_files, Config, sonic_layer_depth, section_cfield

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)

r_bound = (7.5e3, 47.5e3)
tl_z_bounds = (150., 250.)

test_field = 'bg'
tl_file = tl_files[15]

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']

tl_data = np.load(tl_file)

z_a_pe = tl_data['zplot']
z_a = tl_data['z_a']
r_a_pe = tl_data['rplot'] - tl_data['xs']
p_bg = tl_data['p_bg']
p_tilt = tl_data['p_tilt']
p_spice = tl_data['p_spice']
p_total = tl_data['p_total']

x_sec, c_field = section_cfield(tl_data['xs'], x_a, fields['c_' + test_field], rmax=cf.rmax)
p_field = tl_data['p_' + test_field]

sld_z, sld_i = sonic_layer_depth(z_a, c_field, z_max=200)
c_sld = c_field[sld_i, range(c_field.shape[1])]
kx = 2 * pi * fc / c_sld

# trace beam through transition layer
z_i = z_a < tl_z_bounds[1]
ssp_transition = c_field[z_i]
# set c values above sld to 0
for i, cf in zip(sld_i, ssp_transition.T): cf[:i+1] = 0

ky = np.sqrt((2 * pi * fc / ssp_transition) ** 2 - kx ** 2)
dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
dx = dz * (kx / ky)

tl_z_i = np.argmin(np.abs(z_a - tl_z_bounds[0]))
pre_distance = np.sum(dx[:tl_z_i, :], axis=0)
tl_distance = np.sum(dx[tl_z_i:, :], axis=0)

#Energy in mixed layer and transition layer computed by PE
z_ml_pe = z_a_pe < tl_z_bounds[0]
p_ml = p_field[:, z_ml_pe]
en_ml = np.sum(np.abs(p_ml ** 2), axis=1) * dz

z_tl_pe = (z_a_pe > tl_z_bounds[0]) & (z_a_pe < tl_z_bounds[1])
p_tl = p_field[:, z_tl_pe]
en_tl = np.sum(np.abs(p_tl ** 2), axis=1) * dz

en_model = np.zeros(en_ml.size - 1)
en_model = en_ml[:-1] * r_a_pe[:-1]
en_model /= r_a_pe[1:]
tl_in = en_model - en_ml[1:]
tl_in[tl_in < 0] = 0

dr = (r_a_pe[-1] - r_a_pe[0]) / (r_a_pe.size - 1)

pd_up = np.interp(r_a_pe, x_sec, pre_distance)
tl_up = np.interp(r_a_pe, x_sec, tl_distance)
tl_model = np.zeros(r_a_pe.size)
for i, (pd, td, p_in) in enumerate(zip(pd_up, tl_up, tl_in)):
    #start_i = i + int(pd / dr) + 1
    start_i = i + 1
    if start_i >= r_a_pe.size - 1:
        break
    end_i = min(start_i + int(td / dr), r_a_pe.size - 1)
    tl_model[start_i: end_i] += tl_in[i] * r_a_pe[start_i] / r_a_pe[start_i: end_i]

fig, ax = plt.subplots()
ax.plot(r_a_pe / 1e3, 10 * np.log10(en_tl * r_a_pe))
ax.plot(r_a_pe / 1e3, 10 * np.log10(tl_model * r_a_pe))
ax.set_ylim(-50, -0)
