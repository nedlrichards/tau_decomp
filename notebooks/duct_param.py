import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from src import sonic_layer_depth, section_cfield

plt.ion()
plt.style.use('elr')

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']

def fit_cslope(c_field, sld_i):
    """linear regression fit of slope along transcet"""
    m = []
    b = []
    err = []
    for c, i in zip(c_field, sld_i):
        fit = linregress(z_a[:i], c[:i])
        m.append(fit.slope)
        b.append(fit.intercept)
        err.append(fit.stderr)
    m = np.array(m)
    b= np.array(b)
    err = np.array(err)
    return m, b, err

sld = {'bg':sonic_layer_depth(z_a, c_bg, z_max=300),
       'spice':sonic_layer_depth(z_a, c_spice, z_max=300),
       'tilt':sonic_layer_depth(z_a, c_tilt, z_max=300),
       'total':sonic_layer_depth(z_a, c_total, z_max=300)}

rgs = {'bg':fit_cslope(c_bg.T, sld['bg'][1]),
       'tilt':fit_cslope(c_tilt.T, sld['tilt'][1]),
       'spice':fit_cslope(c_spice.T, sld['spice'][1]),
       'total':fit_cslope(c_total.T, sld['total'][1])}

fig, axes = plt.subplots(3, 1, sharex=True)

ax = axes[0]
ax.plot(x_a / 1e3, rgs['total'][0])
ax.plot(x_a / 1e3, rgs['tilt'][0])
ax.plot(x_a / 1e3, rgs['spice'][0])
ax.plot(x_a / 1e3, rgs['bg'][0])
ax.set_ylim(0, 0.06)

ax = axes[1]
ax.plot(x_a / 1e3, sld['total'][0])
ax.plot(x_a / 1e3, sld['tilt'][0])
ax.plot(x_a / 1e3, sld['spice'][0])
ax.plot(x_a / 1e3, sld['bg'][0])
ax.set_ylim(125, 25)

ax = axes[2]
ax.plot(x_a / 1e3, rgs['total'][2])
ax.plot(x_a / 1e3, rgs['tilt'][2])
ax.plot(x_a / 1e3, rgs['spice'][2])
ax.plot(x_a / 1e3, rgs['bg'][2])
ax.set_ylim(0., 0.002)

ax.set_xlim(230, 290)

p_i = np.argmin(np.abs(x_a - 258e3))
fig, ax = plt.subplots()
ax.plot(c_total[:, p_i], z_a)
ax.plot(c_tilt[:, p_i], z_a)
ax.plot(c_spice[:, p_i], z_a)

ax.set_prop_cycle(None)

ax.plot(z_a * rgs['total'][0][p_i] + rgs['total'][1][p_i], z_a, '--')
ax.plot(z_a * rgs['tilt'][0][p_i] + rgs['tilt'][1][p_i], z_a, '--')
ax.plot(z_a * rgs['spice'][0][p_i] + rgs['spice'][1][p_i], z_a, '--')

ax.set_ylim(150, 0)
ax.set_xlim(1506, 1512)

dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
dc = np.diff(c_total[:, p_i])

fig, ax = plt.subplots()
ax.plot(dc / dz, z_a[1:])
ax.set_ylim(150, 0)

