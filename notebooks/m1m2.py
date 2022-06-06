import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from src import list_tl_files, Config, MLEnergyPE, section_cfield, RDModes, MLEnergy
import pyducts

plt.ion()
plt.style.use('elr')

fc = 400
source_depth = 'shallow'

tl_files = list_tl_files(fc, source_depth=source_depth)
tl_file = tl_files[50]
mle = MLEnergy(tl_file)


class M1Energy:
    """Compute energy stored in mode 1"""

    def __init__(self, tl_file, field_type):
        """Mode amps and shapes are in tl file"""
        # precomputed integrated normal mode energy
        proc_eng = np.load('data/processed/energy_processing.npz')
        self.cf = Config()

        self.r_a = proc_eng['r_a']
        self.bg_eng_400 = proc_eng["bg_eng_400"]
        self.dynamic_eng_400 = proc_eng["dynamic_eng_400"]

        tl = np.load(tl_files[8])
        self.xs = tl['xs']
        self.z_a = tl['z_a']
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)
        self.z_int = self.z_a[self.z_a < self.cf.z_int]

        self.r_modes = tl['r_modes']
        self.k_bg = tl['k_' + field_type]
        self.psi_bg = tl['psi_' + field_type]
        self.mode_amps_bg = tl[field_type + '_mode_amps']


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

    def ml_energy(self, p_field):
        """Integrate pressure field to estimate energy"""
        en_ml = np.sum(np.abs(p_field) ** 2, axis=1)
        en_ml *= self.dz
        return en_ml


def run_ram(rf, x_a, z_a, cfield):
    """ram is the model of choice"""

    # RD TL from ram
    rf.write_frontmatter()

    xs = x_a[0]
    for x, c_p in zip(x_a, cfield.T):
        rf.write_profile(x - xs, z_a, c_p)

    pyducts.ram.run_ram()

    zplot, rplot, p_ram = pyducts.ram.read_grid()
    return zplot, xs + rplot, p_ram



tl_files = list_tl_files(fc, source_depth=source_depth)
tl_file = tl_files[50]
m1 = M1Energy(tl_file, 'bg')

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']

x_sec, c_bg_sec = section_cfield(m1.xs, x_a, c_bg, rmax=m1.cf.rmax)

rd_modes = RDModes(c_bg_sec, x_sec, m1.z_a, m1.cf)
D = z_a[-1]
rf = pyducts.ram.RamIn(m1.cf.fc, m1.cf.z_src, m1.cf.rmax, D,
                       bottom_HS=m1.cf.bottom_HS,
                       dr=100., zmax_plot=D)
# run ri ram
rf.write_frontmatter()
rf.write_profile(0., z_a, rd_modes.bg_prof)
pyducts.ram.run_ram()
zplot, rplot, p_ram = pyducts.ram.read_grid()

m1_inds = m1.mode1_inds(0.1)
p_bg = m1.synthesize_pressure()
p_bg_m1 = m1.synthesize_pressure(mode_inds=m1_inds)

en_bg = m1.ml_energy(p_bg)
en_bg_m1 = m1.ml_energy(p_bg_m1)

pe = MLEnergyPE(tl_file)
eng_pe = pe.ml_energy('bg')

# ri pe energy
dz = (zplot[-1] - zplot[0]) / (zplot.size - 1)
z_i = zplot < m1.cf.z_int
eng_pe_test = np.sum(np.abs(p_ram[:, z_i]) ** 2, axis=1) * dz

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

cm = axes[0].pcolormesh(m1.r_a / 1e3, m1.z_int,
                        20 * np.log10(np.abs(p_bg)).T,
                        cmap=m1.cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[1].pcolormesh(m1.r_a / 1e3, m1.z_int,
                        20 * np.log10(np.abs(p_bg_m1)).T,
                        cmap=m1.cf.cmap, vmax=-50, vmin=-90, rasterized=True)
cm = axes[2].pcolormesh(rplot / 1e3, zplot,
                        20 * np.log10(np.abs(p_ram)).T,
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

fig, ax = plt.subplots()
ax.plot(pe.r_a / 1e3, 10 * np.log10(eng_pe * pe.r_a))
ax.plot(m1.r_a / 1e3, 10 * np.log10(en_bg * m1.r_a))
ax.plot(m1.r_a / 1e3, 10 * np.log10(en_bg_m1 * m1.r_a))
ax.plot(rplot / 1e3, 10 * np.log10(eng_pe_test * rplot))
