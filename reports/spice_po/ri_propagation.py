import numpy as np
from math import pi
import matplotlib.pyplot as plt
from os.path import join

from src import RDModes, Config

from src import MLEnergy, list_tl_files

plt.ion()

fc = 400
tl_files = list_tl_files(fc)
ml_modes = MLEnergy(fc, tl_files[23])

modes = ml_modes.field_modes['bg']
psi_s = np.exp(1j * pi / 4) / (modes.rho0 * np.sqrt(8 * pi)) \
      * modes.psi_ier(modes.z_src)
psi_s /= np.sqrt(modes.k_bg)
psi_s *= 4 * pi
# reference ml energy
p_ri = modes.synthesize_pressure(psi_s, ml_modes.z_a, r_synth=ml_modes.r_a)
en_ri = np.sum(np.abs(p_ri) ** 2, axis=1) * ml_modes.dz

p_dB = 20 * np.log10(np.abs(p_ri))
r_a = (ml_modes.r_a + ml_modes.xs) / 1e3,

fig, ax = plt.subplots()
cm = ax.pcolormesh(r_a,  ml_modes.z_a, p_dB.T, cmap=ml_modes.cf.cmap,
                   vmin=-90, vmax=-50)
cb = fig.colorbar(cm)
ax.set_ylim(150, 0)
ax.set_xlabel('Position, $x$ (km)')
ax.set_ylabel('Depth, (m)')
cb.set_label('Pressure re 1 m (dB)')

pos = ax.get_position()
pos.x0 += 0.02
pos.x1 += 0.04
pos.y1 += 0.05
pos.y0 += 0.05
ax.set_position(pos)

pos = cb.ax.get_position()
pos.x0 += 0.02
pos.x1 += 0.02
pos.y1 += 0.05
pos.y0 += 0.05
cb.ax.set_position(pos)

savedir = 'reports/spice_po/figures'
fig.savefig(join(savedir, 'ri_xmission.png'), dpi=300)
