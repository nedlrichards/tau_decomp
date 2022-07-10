import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import os

from src import MLEnergy, MLEnergyPE, list_tl_files, Config

plt.ion()
plt.style.use('elr')

fc = 400
source_depth = 'shallow'
tl_files = list_tl_files(fc=fc)

cf = Config(fc=fc, source_depth=source_depth)

percent_max = 99.99

def load_m1(tl_file):
    tl_data = np.load(tl_file)
    r_a = tl_data['rplot']

    eng_mode = MLEnergy(tl_file, m1_percent=percent_max)
    m1_amps = []
    for field_type in cf.field_types:
        ind = eng_mode.set_1[field_type]
        m1_amps.append(np.squeeze(tl_data[field_type + "_mode_amps"][:, ind]))
    return np.array(m1_amps)

m1_amps = []
for tl_file in tl_files:
    m1_amps.append(load_m1(tl_file))
m1_amps = np.array(m1_amps)

tl_data = np.load(tl_files[0])
r_a = tl_data['rplot']
z_a = tl_data['z_a']

amps_dB = 20 * np.log10(np.abs(m1_amps))
#amps_dB -= amps_dB[:, :, :1]

test_i = 10
field_type = 'bg'
tl_data = np.load(tl_file)
p_field = tl_data['p_' + field_type]
z_a = tl_data['zplot']
r_a = tl_data['rplot']

r_test = r_a[0] - tl_data['xs']

eng_mode = MLEnergy(tl_file, m1_percent=percent_max)
ind = eng_mode.set_1[field_type]
psi_1 = eng_mode.field_modes[field_type].psi_bg[ind]

psi_ier = interp1d(tl_data['z_a'], np.squeeze(psi_1))
psi_proj = psi_ier(z_a)
dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
z0_ind = np.argmax(np.abs(np.diff(np.sign(psi_proj))) > 1.5)

total_eng = np.sum(np.abs(p_field[:, :z0_ind]) ** 2, axis=-1) * dz / 1e3
proj_amp = np.sum(p_field[:, :z0_ind] * psi_proj[:z0_ind], axis=-1) * dz / 1e3

fig, ax = plt.subplots()
#ax.plot(r_a / 1e3, total_eng)
ax.plot(r_a / 1e3, np.abs(proj_amp) * np.sqrt(r_a - tl_data['xs']))
#ax.plot(r_a / 1e3, np.abs(proj_amp))
ax.plot(r_a / 1e3, np.abs(m1_amps[test_i, 0, :]))



cmap = plt.cm.cividis
i_co = 30
clrs = np.ones(amps_dB.shape[0]) * 0.85
clrs[i_co:] = 0.5
alpha = np.ones(amps_dB.shape[0])
#alpha[i_co:] = 0.4
clrs = cmap(clrs)

# background amps
fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.5))

for i, (e, c, a) in enumerate(zip(amps_dB[:, 0, :], clrs, alpha)):
    ax.plot(r_a / 1e3, e, color=c, alpha=a)

cax = fig.add_axes([0.27, 0.3, .02, .1])
lcmap = ListedColormap([clrs[0], clrs[-1]])
sm = ScalarMappable(cmap=lcmap)
cb = fig.colorbar(sm, cax=cax, ticks=[0.25, 0.75])
cb.set_ticklabels([r'$x_{src}\leq$ 300 km', '$x_{src}$ >300km'])

ax.set_xlim(r_a[0] / 1e3, r_a[-1] / 1e3)

ax.set_xlabel('Range (km)')
ax.set_ylabel('Mode 1 magnitude (dB re 1 m)')

pos = ax.get_position()
pos.x0 += 0.10
pos.x1 += 0.05
pos.y0 += 0.08
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/bg_m1_amp.png', dpi=300)

