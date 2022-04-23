import numpy as np
from math import pi
import os

from src import RDModes, Config, list_tl_files
import matplotlib.pyplot as plt

plt.ion()

fc = 400
z_int = 150.
cf = Config(fc)

tl_files = list_tl_files(fc)
tl_data = np.load(tl_files[23])

r_a = tl_data['rplot']
rd_modes = RDModes(tl_data['c_bg'], tl_data['x_a'], tl_data['z_a'],
                    cf.fc, cf.z_src, s=None)

xs = tl_data['xs']
dr = (rd_modes.r_plot[-1] - rd_modes.r_plot[0]) / (rd_modes.r_plot.size - 1)
r_max = 60e3
num_r = int(np.ceil(r_max / dr))
r_a_modes = (np.arange(num_r) + 1) * dr

l_len = -2 * pi / (np.diff(np.real(rd_modes.k_bg)) - np.spacing(1))

# reference energy
psi_s = np.exp(1j * pi / 4) / (rd_modes.rho0 * np.sqrt(8 * pi)) \
        * rd_modes.psi_ier(rd_modes.z_src)
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

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(cf.jasa_1clm, 2.75))
ax[0].plot(rd_modes.bg_prof, rd_modes.z_a, color='0.2')
ax[1].plot(rd_modes.psi_bg[am[0], :], rd_modes.z_a, label='#'+str(labels[0]))
ax[1].plot(rd_modes.psi_bg[am[1], :], rd_modes.z_a, label='#'+str(labels[1]))
ax[1].plot(rd_modes.psi_bg[am[2], :], rd_modes.z_a, label='#'+str(labels[2]))
ax[0].set_xlim(1497, 1510)
ax[1].set_ylim(150, 0)
ax[0].grid()
ax[0].set_xticks([1505], minor=True)
ax[1].grid()
ax[0].set_xlabel('Sound speed (m/s)')
ax[1].set_xlabel('Mode amplitude')
ax[0].set_ylabel('Depth (m)')
ax[1].legend(loc=(0.48, 0.02), framealpha=1)

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

