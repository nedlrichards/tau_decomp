import numpy as np
from math import pi
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.linalg import solve
from scipy.signal import find_peaks
from os import path

from src import sonic_layer_depth

import pyducts
from modepy import PRModes


class RDModes:
    """compute range dependent mode coupling for a gaussian pertubation"""

    def __init__(self, c_field, x_a, z_a, config, r_decimation=10, psi_k_bg=None):
        """Setup position and verticle profiles for gaussian"""
        self.cf = config
        self.c_field = c_field

        self.bg_prof = np.mean(c_field, axis=1)
        self.dc = c_field - self.bg_prof[:, None]
        # depth of climatology
        self.lim_i = int(np.median(np.argmax(self.dc == 0, axis=0)))

        self.z_a = z_a
        self.dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

        self.x_start = x_a[0]
        self.x_a = x_a
        self.dx = (x_a[-1] - x_a[0]) / (x_a.size - 1)
        self.r_prof = x_a - x_a[0]
        self.r_decimation = r_decimation

        self.plot_dx = self.dx / self.r_decimation
        num_steps = self.r_prof[-1] / self.plot_dx
        self.r_plot = (np.arange(num_steps) + 1) * self.plot_dx

        self.c0 = np.mean(self.bg_prof)
        self.omega = 2 * pi * self.cf.fc
        self.k0 = self.omega / self.c0

        self.rho0 = 1000.
        self.run_file = path.join('envs', f'auto_gen')
        #self.s = s

        if psi_k_bg is None:
            #psi_bg, self.k_bg = self.run_kraken_bg()
            modes = PRModes(self.z_a, self.bg_prof, self.cf.fc)
            self.k_bg, psi_bg = modes(c_bounds=self.cf.c_bounds)
        else:
            psi_bg, self.k_bg = psi_k_bg

        self.psi_bg = np.real(psi_bg)
        self.bg_sld, sld_i = sonic_layer_depth(z_a, self.bg_prof[:, None], z_max=300.)

        self.psi_ier = interp1d(self.z_a, self.psi_bg)
        self.k_cross = np.real(self.k_bg)[:, None] * np.real(self.k_bg)[None, :]
        self.k_diff = self.k_bg[:, None] - self.k_bg[None, :]

        # mode number calculation
        # using with num zero crossings, could try num maxima
        ml_modes = self.psi_bg[:, :sld_i[0]]
        ml_modes[np.abs(ml_modes) < 1e-10] = 0

        crossings = np.sign(ml_modes)
        crossings = np.abs(np.diff(crossings, axis=1))[:, 1:] // 2
        crossings = np.sum(crossings, axis=1)

        # identify evanecent modes
        zero_c = (crossings == 0)
        mode_test = self.psi_bg[zero_c, :][:, :sld_i[0]]
        peaks = [find_peaks(m, prominence=0.1)[0] for m in mode_test]
        eva_i = np.where(np.array([p.size < 1 for p in peaks]))[0]

        self.mode_number = crossings
        self.mode_number[eva_i] = -1

        self.rho_scale = self.k0 ** 2 * self.dz \
                       / (self.rho0 * np.sqrt(self.k_cross))

        self.dux_rd = None

    def rho(self, r):
        """definition of rho taken from Colosi and Morozov 2009, Eq. 3"""
        r_i = int(r // self.dx)
        mu = self.dc[:, r_i] / self.c0
        # only integrate above climatology
        mu = mu[: self.lim_i]
        psi = self.psi_bg[:, :self.lim_i]
        integration = (psi * mu) @ psi.T
        rho = self.rho_scale * integration
        return rho


    def run_kraken_bg(self):
        """Run kraken to compute modes"""
        dux = pyducts.modes.Kraken(self.run_file, 100., self.z_a,
                                   c_bounds=self.cf.c_bounds)
        if self.s is not None:
            bg_ier = UnivariateSpline(self.z_a, self.bg_prof, k=1, s=self.s)
            dux.write_env(self.cf.fc,
                        bg_ier.get_knots(),
                        bg_ier.get_coeffs(),
                        bottom_HS=self.cf.bottom_HS)
        else:
            dux.write_env(self.cf.fc,
                        self.z_a,
                        self.bg_prof,
                        bottom_HS=self.cf.bottom_HS)

        dux.run_kraken()
        psi_bg, k_bg, _ = pyducts.modes.read_mod(self.run_file)

        return psi_bg, k_bg

    def run_kraken_cp(self):
        """Compute coupled mode result with kraken"""

        # run kraken, range dependent
        rf = self.run_file + "_rd"
        dux_rd = pyducts.modes.Kraken(rf, self.r_plot, self.z_a,
                                      z_src=self.cf.z_src, c_bounds=self.cf.c_bounds)

        for i, prof in enumerate(self.c_field.T):
            # resample profiles to compute delta
            if self.s is not None:
                dc_ier = UnivariateSpline(self.z_a, prof, k=1, s=self.s)
                prof = [dc_ier.get_knots(), dc_ier.get_coeffs()]
            else:
                prof = [self.z_a, prof]

            if i == 0:
                dux_rd.write_env(self.cf.fc, prof[0], prof[1],
                              append=False, bottom_HS=self.cf.bottom_HS)
            else:
                dux_rd.write_env(self.cf.fc, prof[0], prof[1],
                              append=True, bottom_HS=self.cf.bottom_HS)

        dux_rd.run_kraken()
        self.dux_rd = dux_rd


    def pressure_kraken_cp(self):
        """
        compute pressure from kraken
        """
        # mode coupling
        self.dux_rd.run_field(raxis=self.r_prof, option='RC')
        z_plot, r_plot, p_krak_rd = pyducts.modes.read_shd(self.run_file + "_rd")

        return z_plot, r_plot, p_krak_rd


    def couple_cn(self):
        """Direct Crank-Nickolson solution, coupled modes"""
        # setup source term
        phi_s = np.exp(1j * pi / 4) / (self.rho0 * np.sqrt(8 * pi)) \
              * self.psi_ier(self.cf.z_src)

        a_cn = [phi_s]
        rho_last = self.rho(self.r_plot[0])
        exp_last = np.exp(1j * self.k_diff * self.r_plot[0])

        ident = np.identity(self.psi_bg.shape[0], dtype=np.complex128)

        for r in self.r_plot[1:]:
            rho_current = self.rho(r)
            exp_current = np.exp(1j * self.k_diff * r)

            A_n1 = (self.plot_dx * 1j / 2) * rho_current * exp_current
            lhs = ident + A_n1.T

            A_n = (self.plot_dx * 1j / 2) * rho_last * exp_last
            rhs = ident - A_n.T

            a_next = solve(lhs, rhs @ a_cn[-1])
            a_cn.append(a_next)

            rho_last = rho_current
            exp_last = exp_current

        a_cn = np.array(a_cn)

        # normalize amplitudes
        a_cn /= np.sqrt(self.k_bg[None, :])
        a_cn *= 4 * pi
        return a_cn


    def synthesize_pressure(self, amps, z_rcr, r_synth=None):
        """synthesize pressure from modal amplitudes"""
        if len(amps.shape) == 1:
            amps = amps[None, :, None]
        else:
            amps = amps[:, :, None]

        # formulation follows Colosi and Morozov 2009
        z_rcr = np.array(z_rcr, ndmin=1)
        psi_rcr = self.psi_ier(z_rcr)[None, :, :]

        if r_synth is None:
            r_synth = self.r_plot
        elif amps.shape[0] > 1:
            amp_ier = interp1d(self.r_plot, amps[:, :, 0].T,
                               bounds_error=False,
                               fill_value=(amps[0, :, 0], amps[-1, :, 0]))
            amps = amp_ier(r_synth).T
            amps = amps[:, :, None]

        pressure = amps * psi_rcr * np.exp(1j
                                           * self.k_bg[None, :, None]
                                           * r_synth[:, None, None])
        pressure /= np.sqrt(r_synth[:, None, None])
        pressure = pressure.sum(axis=1)

        return pressure
