import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from src import list_tl_files, Config

plt.ion()
plt.style.use('elr')

fc = 400
source_depth = 'shallow'


class M1Energy:
    """Compute energy stored in mode 1"""

    def __init__(self, tl_file):
        """Mode amps and shapes are in tl file"""
        # precomputed integrated normal mode energy
        proc_eng = np.load('data/processed/energy_processing.npz')
        self.cf = Config()

        self.r_a = proc_eng['r_a']
        self.bg_eng_400 = proc_eng["bg_eng_400"]
        self.dynamic_eng_400 = proc_eng["dynamic_eng_400"]

        tl = np.load(tl_files[8])
        self.z_a = tl['z_a']
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)
        self.z_int = self.z_a[self.z_a < self.cf.z_int]

        self.r_modes = tl['r_modes']
        self.k_bg = tl['k_bg']
        self.psi_bg = tl['psi_bg']
        self.mode_amps_bg = tl['bg_mode_amps']


    def mode1_inds(self, percent_max):
        """Compute mode 1 indices"""

        ll = 2 * pi / (self.k_bg[:-1] - self.k_bg[1:])
        peaks = find_peaks(ll)[0]
        peaks = peaks[np.argsort(ll[peaks])][::-1]

        m1_ind = peaks[0]

        z_i = self.z_a < self.cf.z_int
        search_i = np.arange(m1_ind - 2, m1_ind + 3)

        mode_eng = np.sum(self.psi_bg[m1_ind - 2: m1_ind + 3, z_i] ** 2, axis=-1)
        mode_eng /= np.max(mode_eng)

        mode1 = np.where(mode_eng > percent_max)[0] + search_i[0]
        return mode1


    def synthesize_pressure(self, mode_inds=None):
        """synthesize pressure from modal amplitudes"""
        if mode_inds is None:
            mode_inds = np.arange(self.mode_amps_bg.shape[-1])

        amps = self.mode_amps_bg[:, mode_inds, None]

        # formulation follows Colosi and Morozov 2009
        psi_ier = interp1d(self.z_a, self.psi_bg[mode_inds, :])
        psi_rcr = psi_ier(self.z_int)[None, :, :]

        pressure = amps * psi_rcr * np.exp(1j
                                           * self.k_bg[None, mode_inds, None]
                                           * self.r_a[:, None, None])

        pressure /= np.sqrt(self.r_a[:, None, None])
        pressure = pressure.sum(axis=1)

        return pressure

tl_files = list_tl_files(fc, source_depth=source_depth)
m1 = M1Energy(tl_files[10])

m1_inds = m1.mode1_inds(0.1)
p_bg = m1.synthesize_pressure()
p_bg_m1 = m1.synthesize_pressure(mode_inds=m1_inds)

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

cm = axes[0].pcolormesh(m1.r_a / 1e3, m1.z_int,
                        20 * np.log10(np.abs(p_bg)).T,
                        cmap=m1.cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[1].pcolormesh(m1.r_a / 1e3, m1.z_int,
                        20 * np.log10(np.abs(p_bg_m1)).T,
                        cmap=m1.cf.cmap, vmax=-50, vmin=-90, rasterized=True)

#fig.supxlabel('Range (km)')
#fig.supylabel('Depth (m)')
axes[0].set_ylim(m1.cf.z_int, 0)
axes[0].set_xlim(m1.r_a[0] / 1e3, m1.r_a[-1] / 1e3)

pos = axes[0].get_position()
pos.x0 -= 0.01
pos.x1 += 0.05
pos.y0 -= 0.02
pos.y1 -= 0.02
axes[0].set_position(pos)




