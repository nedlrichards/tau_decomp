import numpy as np
from math import pi
from scipy.signal import find_peaks

from src import RDModes, Config

class MLEnergyPE:
    """Simple energy calculations from PE result"""
    def __init__(self, fc, run_file):
        """Calculate range independent modes"""
        self.cf = Config(fc)
        self.tl_data = np.load(run_file)

        self.xs = self.tl_data['xs']
        self.r_a = self.tl_data['rplot'] - self.xs
        self.z_a = self.tl_data['zplot']
        self.z_i = self.z_a < self.cf.z_int
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)

    def ml_energy(self, field_type):
        """energy from pe"""
        p_ml = self.tl_data['p_' + field_type][:, self.z_i] ** 2
        en_pe = np.sum(np.abs(p_ml), axis=1)
        en_pe *= self.dz
        return en_pe

class MLEnergy:
    """Different methods to calculate or estimate mixed layer energy"""

    def __init__(self, run_file):
        """Calculate range independent modes"""
        self.tl_data = np.load(run_file)
        self.cf = Config(self.tl_data['fc'][()])

        bg_modes = RDModes(self.tl_data['c_bg'],
                           self.tl_data['x_a'],
                           self.tl_data['z_a'],
                           self.cf.fc, self.cf.z_src,
                           c_bounds=self.cf.c_bounds, s=None)

        tilt_modes = RDModes(self.tl_data['c_tilt'],
                             self.tl_data['x_a'],
                             self.tl_data['z_a'],
                             self.cf.fc, self.cf.z_src,
                             c_bounds=self.cf.c_bounds, s=None)

        spice_modes = RDModes(self.tl_data['c_spice'],
                              self.tl_data['x_a'],
                              self.tl_data['z_a'],
                              self.cf.fc, self.cf.z_src,
                              c_bounds=self.cf.c_bounds, s=None)

        total_modes = RDModes(self.tl_data['c_total'],
                              self.tl_data['x_a'],
                              self.tl_data['z_a'],
                              self.cf.fc, self.cf.z_src,
                              c_bounds=self.cf.c_bounds, s=None)

        self.field_modes = {'bg':bg_modes, 'tilt':tilt_modes,
                            'spice':spice_modes, 'total':total_modes}

        # common axes
        self.xs = self.tl_data['xs']
        self.r_a = self.tl_data['rplot'] - self.xs
        self.z_a = self.tl_data['zplot']
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)
        self.z_i = self.z_a < self.cf.z_int

        # loop lengths
        eps = np.spacing(1)
        bg_llen = -2 * pi / (np.diff(np.real(bg_modes.k_bg)) - eps)
        tilt_llen = -2 * pi / (np.diff(np.real(tilt_modes.k_bg)) - eps)
        spice_llen = -2 * pi / (np.diff(np.real(spice_modes.k_bg)) - eps)
        total_llen = -2 * pi / (np.diff(np.real(total_modes.k_bg)) - eps)
        self.llen = {'bg':bg_llen, 'tilt':tilt_llen,
                     'spice':spice_llen, 'total':total_llen}

        bg_set_1 = self.mode_set_1(self.llen['bg'])
        tilt_set_1 = self.mode_set_1(self.llen['tilt'])
        spice_set_1 = self.mode_set_1(self.llen['spice'])
        total_set_1 = self.mode_set_1(self.llen['total'])
        self.set_1 = {'bg':bg_set_1, 'tilt':tilt_set_1,
                     'spice':spice_set_1, 'total':total_set_1}

        bg_set_2 = self.mode_set_2(self.llen['bg'], bg_set_1)
        tilt_set_2 = self.mode_set_2(self.llen['tilt'], tilt_set_1)
        spice_set_2 = self.mode_set_2(self.llen['spice'], spice_set_1)
        total_set_2 = self.mode_set_2(self.llen['total'], total_set_1)
        self.set_2 = {'bg':bg_set_2, 'tilt':tilt_set_2,
                     'spice':spice_set_2, 'total':total_set_2}


    def field_ml_eng(self, field_type, indicies=None):
        """Compute pressure from one field type"""
        # reduced mode set estimate of energy
        psi_rd = self.tl_data[field_type + '_mode_amps'].copy()
        rd_modes = self.field_modes[field_type]

        if indicies is not None:
            psi_0 = np.zeros_like(psi_rd)
            psi_0[:, indicies] = psi_rd[:, indicies]
            p_rd = rd_modes.synthesize_pressure(psi_0,
                                                self.z_a,
                                                r_synth=self.r_a)
            en_rd = np.sum(np.abs(p_rd) ** 2, axis=1) * self.dz
        else:
            p_rd = rd_modes.synthesize_pressure(psi_rd,
                                                self.z_a,
                                                r_synth=self.r_a)
            en_rd = np.sum(np.abs(p_rd) ** 2, axis=1) * self.dz

        return en_rd


    def background_diffraction(self):
        # range independent mode amplitudes
        modes = self.field_modes['bg']
        psi_s = np.exp(1j * pi / 4) / (modes.rho0 * np.sqrt(8 * pi)) \
                * modes.psi_ier(modes.z_src)
        psi_s /= np.sqrt(modes.k_bg)
        psi_s *= 4 * pi
        # reference ml energy
        p_ri = modes.synthesize_pressure(psi_s, self.z_a, r_synth=self.r_a)
        en_ri = np.sum(np.abs(p_ri) ** 2, axis=1) * self.dz

        # resticted mode calculation
        psi_m0 = np.zeros_like(psi_s)
        psi_m0[self.set_1['bg']] = psi_s[self.set_1['bg']]
        p_m0 = modes.synthesize_pressure(psi_m0, self.z_a, r_synth=self.r_a)
        en_ri_0 = np.sum(np.abs(p_m0) ** 2, axis=1) * self.dz

        return en_ri, en_ri_0


    def mode_set_1(self, llen):
        """Common calculation of mode set 1 from loop length"""
        am = np.argmax(llen)
        dom_modes = np.zeros(llen.size, dtype=np.bool_)

        if llen[am + 1] > 6e4:
            am = [am, am + 1]
        else:
            am = [am]

        am = np.hstack([[am[0] - 1], am, [am[-1] + 1]])
        dom_modes[am] = True
        return list(np.where(dom_modes)[0])

    def mode_set_2(self, llen, m1):
        """Common calculation of mode set 2 from loop length"""
        # mode 2 extends mode 1 out to minimum after 2nd peak
        maxs = find_peaks(llen)[0]
        maxs = maxs[np.argsort(llen[maxs])]
        mins = find_peaks(-llen)[0]
        p2 = maxs[-2]
        t2 = mins[mins > p2][0]

        m2 = m1.copy()
        m2 += list(range(m2[-1] + 1, t2 + 1))

        return m2
