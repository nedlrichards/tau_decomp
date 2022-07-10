import numpy as np
from math import pi
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from src import RDModes, Config, section_cfield

class MLEnergy:
    """Different methods to calculate or estimate mixed layer energy"""

    def __init__(self, run_file, source_depth="shallow"):
        """Calculate range independent modes"""
        self.tl_data = np.load(run_file)
        self.cf = Config(source_depth=source_depth,
                         fc=self.tl_data['fc'][()],
                         c_bounds=[1503., 1525.])

        # common axes
        self.xs = self.tl_data['xs']
        self.r_a = self.tl_data['rplot'] - self.xs
        self.z_a = self.tl_data['zplot']
        self.z_i = self.z_a < self.cf.z_int
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)

        self.field_modes = {}
        self.llen = {}
        self.set_1 = {}

        self._start_field_type('bg')
        self._start_field_type('tilt')
        self._start_field_type('spice')
        self._start_field_type('total')


    def _start_field_type(self, field_type):
        """Common startup by field type"""
        decomp = np.load(self.cf.decomp_npz)
        c_total = decomp['c_' + field_type]
        x_a = decomp['x_a']

        x_sec, c_sec = section_cfield(self.xs, x_a, c_total)
        if "psi_" + field_type in self.tl_data:
            psi_k = (self.tl_data["psi_" + field_type],
                    self.tl_data["k_" + field_type])
        else:
            psi_k = None

        modes = RDModes(c_sec, x_sec, self.tl_data['z_a'], self.cf,
                         psi_k_bg=psi_k)
        self.field_modes[field_type] = modes

        llen = -2 * pi / (np.diff(np.real(modes.k_bg)))
        self.llen[field_type] = llen

        set_1 = self.mode_set_1(field_type)
        self.set_1[field_type] = set_1


    def ml_energy(self, field_type):
        """Energy from mixed layer computed by PE"""
        p_ml = self.tl_data['p_' + field_type][:, self.z_i]
        en_pe = np.sum(np.abs(p_ml ** 2), axis=1)
        en_pe *= self.dz
        return en_pe


    def proj_mode1(self, field_type):
        """Project pressure field onto mode 1 up to 1st zero crossing"""
        m1_ind = self.mode_set_1(field_type, m1_percent=99.9)
        psi_1 = self.field_modes[field_type].psi_bg[m1_ind]

        psi_ier = interp1d(self.tl_data['z_a'], np.squeeze(psi_1))
        psi_proj = psi_ier(self.z_a)
        z0_ind = np.argmax(np.abs(np.diff(np.sign(psi_proj))) > 1.5)

        p_ml = self.tl_data['p_' + field_type][:, :z0_ind]

        proj_amp = np.sum(p_ml * psi_proj[None, :z0_ind], axis=-1) * self.dz
        proj_amp *= np.sqrt(self.r_a / 1e3)

        return proj_amp


    def mode_energy(self, field_type, indicies=None):
        """Compute pressure from one field type"""
        # reduced mode set estimate of energy
        psi_rd = self.tl_data[field_type + '_mode_amps'].copy()
        rd_modes = self.field_modes[field_type]

        if indicies is not None:
            psi_0 = np.zeros_like(psi_rd)
            psi_0[:, indicies] = psi_rd[:, indicies]
            psi_rd = psi_0

        p_rd = rd_modes.synthesize_pressure(psi_rd,
                                            self.z_a[self.z_i],
                                            r_synth=self.r_a)
        en_rd = np.sum(np.abs(p_rd) ** 2, axis=1) * self.dz

        return en_rd


    def background_diffraction(self, field_type):
        """Background energy loss computed with range independent modes"""
        modes = self.field_modes[field_type]
        psi_s = np.exp(1j * pi / 4) / (modes.rho0 * np.sqrt(8 * pi)) \
                * modes.psi_ier(modes.cf.z_src)
        psi_s /= np.sqrt(modes.k_bg)
        psi_s *= 4 * pi

        # reference ml energy
        p_ri = modes.synthesize_pressure(psi_s,
                                         self.z_a[self.z_i],
                                         r_synth=self.r_a)
        en_ri = np.sum(np.abs(p_ri) ** 2, axis=1) * self.dz

        # resticted mode calculation
        psi_m0 = np.zeros_like(psi_s)
        psi_m0[self.set_1[field_type]] = psi_s[self.set_1[field_type]]
        p_m0 = modes.synthesize_pressure(psi_m0,
                                         self.z_a[self.z_i],
                                         r_synth=self.r_a)
        en_ri_0 = np.sum(np.abs(p_m0) ** 2, axis=1) * self.dz

        return en_ri, en_ri_0


    def mode_set_1(self, field_type, m1_percent=10.):
        """Common calculation of mode set 1 from loop length"""
        llen = self.llen[field_type]
        hgt_bnd = np.mean([np.median(llen), np.max(llen)])
        pks = find_peaks(llen, height=hgt_bnd)[0]

        m1_ind = pks[0]
        sld_z = self.field_modes[field_type].bg_sld

        z_i = self.tl_data['z_a'] < sld_z
        search_i = np.arange(m1_ind - 2, m1_ind + 3)

        psi_bg = self.field_modes[field_type].psi_bg

        # test for no zero crossings in ML

        x_test = np.diff(np.sign(psi_bg[search_i, :][:, z_i]))
        is_m1 = ~np.any(np.abs(x_test) > 1.5, axis=-1)

        mode_eng = np.sum(psi_bg[search_i, :][:, z_i] ** 2, axis=-1)
        mode_eng /= np.max(mode_eng)

        is_eng = mode_eng > m1_percent / 100.
        mode1 = np.where(is_m1 & is_eng)[0] + search_i[0]

        return mode1
