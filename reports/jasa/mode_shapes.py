import numpy as np
from math import pi
import os
from scipy.stats import linregress

from src import RDModes, Config, list_tl_files, section_cfield, sonic_layer_depth
import matplotlib.pyplot as plt

plt.style.use('elr')

plt.ion()

fc = 400
z_int = 150.
cf = Config(fc=fc)
# need to get mode number correct
cf.c_bounds = [1000., cf.c_bounds[-1]]

tl_files = list_tl_files(fc)
tl_data = np.load(tl_files[23])
xs = tl_data['xs']

r_a = tl_data['rplot']

fields = np.load('data/processed/inputed_decomp.npz')
x_a, c_bg = section_cfield(xs, fields['x_a'], fields['c_bg'])

rd_modes = RDModes(c_bg, x_a, tl_data['z_a'], cf)

dr = (rd_modes.r_plot[-1] - rd_modes.r_plot[0]) / (rd_modes.r_plot.size - 1)
r_max = 60e3
num_r = int(np.ceil(r_max / dr))
r_a_modes = (np.arange(num_r) + 1) * dr

l_len = -2 * pi / (np.diff(np.real(rd_modes.k_bg)) - np.spacing(1))

# reference energy
psi_s = np.exp(1j * pi / 4) / (rd_modes.rho0 * np.sqrt(8 * pi)) \
        * rd_modes.psi_ier(cf.z_src)
psi_s /= np.sqrt(rd_modes.k_bg)
psi_s *= 4 * pi

z_a = tl_data['zplot']
dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

dom_modes = (rd_modes.mode_number == 0) | (rd_modes.mode_number == 1)
# either 3 or 4 selected modes
dom_modes = np.zeros_like(dom_modes)
am = np.argmax(l_len)

if l_len[am + 1] > 6e4:
    am = [am, am + 1]
else:
    am = [am]

am = np.hstack([[am[0] - 1], am, [am[-1] + 1]])
labels = np.arange(rd_modes.mode_number.size)[am]

z_sld, sld_i = sonic_layer_depth(rd_modes.z_a,
                                 rd_modes.bg_prof[:, None],
                                 z_max=300)
lrg = linregress(rd_modes.z_a[:sld_i[0]], rd_modes.bg_prof[:sld_i[0]])

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(cf.jasa_1clm, 2.75))
ax[0].plot(rd_modes.bg_prof, rd_modes.z_a, color='0.2')
ax[0].plot(rd_modes.z_a * lrg.slope + lrg.intercept, rd_modes.z_a, '--', color="b")
ax[1].plot(rd_modes.psi_bg[am[0], :], rd_modes.z_a, label='#'+str(labels[0]))
ax[1].plot(rd_modes.psi_bg[am[1], :], rd_modes.z_a, label='#'+str(labels[1]))
ax[1].plot(rd_modes.psi_bg[am[2], :], rd_modes.z_a, label='#'+str(labels[2]))
ax[0].set_xlim(1490, 1510)
ax[1].set_ylim(350, 0)
ax[0].grid()
ax[0].set_xticks([1505], minor=True)
ax[1].grid()
ax[0].set_xlabel('Sound speed, $c$ (m/s)')
ax[1].set_xlabel('Mode amplitude')
ax[0].set_ylabel('Depth (m)')
ax[1].legend(loc=(0.48, 0.02), framealpha=1, handlelength=1)

pos = ax[0].get_position()
pos.x0 += 0.05
pos.x1 += -0.02
pos.y1 += 0.06
pos.y0 += 0.06
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += -0.06
pos.x1 += 0.06
pos.y1 += 0.06
pos.y0 += 0.06
ax[1].set_position(pos)

savedir = 'reports/jasa/figures'
fig.savefig(os.path.join(savedir, 'mode_shapes.png'), dpi=300)

print(lrg.slope)
