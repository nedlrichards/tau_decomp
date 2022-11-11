import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os

from src import MLEnergy, list_tl_files, Config, sonic_layer_depth, section_cfield

plt.ion()
plt.style.use('elr')

#fc = 400
fc = 1000
cf = Config(fc=fc)
r_bound = (7.5e3, 37.5e3)

fields = np.load('data/processed/inputed_decomp.npz')
x_field_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']

load_dir = 'data/processed/'
int_eng = np.load(os.path.join(load_dir, 'int_eng_shallow_' + str(int(fc)) + '.npz'))
tester = int_eng['ml_ml'][1:]
tester_tl = int_eng['ml_tl'][1:]

dyn_ml = 10 ** (int_eng['ml_ml'][1:] / 10)
dyn_tl = 10 ** (int_eng['ml_tl'][1:] / 10)

xs = int_eng['xs']
x_pe_a = int_eng['r_a']
dr = (x_pe_a[-1] - x_pe_a[0]) / (x_pe_a.size - 1)
x_i = (r_bound[0] <= x_pe_a) & (r_bound[1] >= x_pe_a)

def ray_lag(i_field, i_xmission, is_mean=True):
    """ray lag is the delay of a ray entering and exiting the transition layer"""

    x_sec, c_field = section_cfield(xs[i_xmission], x_field_a,
                                    fields['c_' + cf.field_types[i_field]],
                                    rmax=cf.rmax)
    if is_mean:
        # mean of c_field
        c_field = np.mean(c_field, axis=1)[:, None] * np.ones((1, c_field.shape[1]))

    sld_z, sld_i = sonic_layer_depth(z_a, c_field, z_max=200)
    c_sld = c_field[sld_i, range(c_field.shape[1])]
    kx = 2 * pi * fc / c_sld

    # trace beam through transition layer
    z_i = z_a < cf.z_tl
    ssp_transition = c_field[z_i]

    # set c values above sld to 0
    for i, c in zip(sld_i, ssp_transition.T): c[:i+1] = 0

    ky = np.sqrt((2 * pi * fc / ssp_transition) ** 2 - kx ** 2)
    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
    dx = dz * (kx / ky)

    tl_z_i = np.argmin(np.abs(z_a - cf.z_ml))
    tl_distance = np.sum(dx[tl_z_i:, :], axis=0)

    return tl_distance

def tl_model(i_field, i_xmission):
    """Predict tl energy for ml loss"""

    en_ml = dyn_ml[i_field, i_xmission, :]
    en_tl = dyn_tl[i_field, i_xmission, :]
    tl_distance = ray_lag(i_field, i_xmission, is_mean=True)

    # predict energy change from spreading alone
    tl_in = -np.diff(en_ml) / x_pe_a[1:]
    tl_in[tl_in < 0] = 0

    # cumulative sum of tl energy
    n = int(tl_distance[0] / dr)
    tl_model = np.cumsum(tl_in, dtype=float)
    tl_model[n:] = tl_model[n:] - tl_model[:-n]
    tl_model = np.concatenate([[tl_model[0]], tl_model])
    return tl_model * x_pe_a

ml_pred = []
for i_f in range(4):
    pred= []
    for i_x in range(dyn_ml.shape[1]):
        pred.append(tl_model(i_f, i_x))
    pred = np.array(pred)
    ml_pred.append(pred)
ml_pred = 10 * np.log10(np.array(ml_pred))


save_dir = 'data/processed/'
sf = os.path.join(save_dir, 'tl_eng_model' + '_' + str(int(fc)) + '.npz')

np.savez(sf, ml_pred=ml_pred)
