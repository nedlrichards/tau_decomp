import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src import list_tl_files

plt.ion()
plt.style.use('elr')

fc = 400
source_depth = 'shallow'

tl_files = list_tl_files(fc, source_depth=source_depth)

tl = np.load(tl_files[10])

z_a = tl['z_a']
r_modes = tl['r_modes']
k_total = tl['k_total']
psi_total = tl['psi_total']
mode_amps_total = tl['total_mode_amps']

ll = 1 * pi / (k_total[:-1] - k_total[1:])
peaks = find_peaks(ll)[0]
peaks = peaks[np.argsort(ll[peaks])][::-1]

m1_ind = peaks[0]
m2_ind = peaks[1]

fig, ax = plt.subplots()
ax.plot(ll)
ax.plot(m1_ind, ll[m1_ind], '.')
ax.plot(m2_ind, ll[m2_ind], '.')

psi_sgn = np.sign(np.diff(psi_total, axis=1))[:, 0]
psi_total *= psi_sgn[:, None]

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(psi_total[m1_ind, :].T, z_a)
ax[0].plot(psi_total[m1_ind - 1, :].T, z_a)
ax[0].plot(psi_total[m1_ind + 1, :].T, z_a)
ax[1].plot(psi_total[m2_ind, :].T, z_a)
ax[1].plot(psi_total[m2_ind - 1, :].T, z_a)
ax[1].plot(psi_total[m2_ind + 1, :].T, z_a)
ax[0].set_ylim(150, 0)

fig, ax = plt.subplots()
ax.plot(r_modes, np.abs(mode_amps_total[:, m1_ind]))
ax.plot(r_modes, np.abs(mode_amps_total[:, m2_ind]))
