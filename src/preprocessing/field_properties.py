import numpy as np
from math import pi
from scipy.io import loadmat
from scipy.interpolate import interp1d
import gsw

from src import Config, SA_CT_from_sigma0_spiciness0

class Field:
    """Generate spice, gamma, in local coordinates"""

    def __init__(self, num_sigma=300, spice_def=1):
        """Standard processing"""
        self.cf = Config()
        grid_data = loadmat('data/processed/stablized_field.mat')
        self.p_ref = 0.
        self.num_sigma = num_sigma
        self.spice_def = spice_def

        self.x_a = np.squeeze(grid_data['x_a']).astype(np.float64) * 1e3
        self.z_a = np.squeeze(grid_data['z_a']).astype(np.float64)

        self.press = gsw.p_from_z(-self.z_a, self.cf.lat)

        # load stabalized properties
        self.xy_sa = (grid_data['SA_stable'].T).astype(np.float64)
        self.xy_ct = (grid_data['CT_stable'].T).astype(np.float64)

        # derived quantities
        self.xy_sig = gsw.sigma0(self.xy_sa, self.xy_ct)
        self.xy_c = gsw.sound_speed(self.xy_sa, self.xy_ct, self.press[:, None])
        self.r_prof = np.broadcast_to(self.x_a, self.xy_sa.shape)

        if self.spice_def == 0:
            print('spiciness')
            self.xy_gamma = gsw.spiciness0(self.xy_sa, self.xy_ct)
            return

        # grid salinity and theta into density bins
        self.sig_bins = np.linspace(self.xy_sig.min(),
                                    self.xy_sig.max(),
                                    num=self.num_sigma + 1)[1:]

        prof_iter = zip(self.xy_sig.T, self.xy_sa.T, self.xy_ct.T)

        props_xsig = []
        for sig_prof, sa_prof, ct_prof in prof_iter:
            i_ct = np.interp(self.sig_bins, sig_prof, ct_prof,
                             left=np.nan, right=np.nan)
            i_sa = np.interp(self.sig_bins, sig_prof, sa_prof,
                             left=np.nan, right=np.nan)
            props_xsig.append(np.array([i_sa, i_ct]))
        props_xsig = np.stack(props_xsig, axis=2)
        self.xsig_sa = props_xsig[0, :, :]
        self.xsig_ct = props_xsig[1, :, :]

        if self.spice_def == 2:
            self.xy_gamma = self.xy_sa
            return


        print('transect average spice definition')
        # compute mean temperature, compute salinity from temperature
        self.sig_mean_sa = np.nanmean(self.xsig_sa, axis=-1)
        self.sig_mean_ct = gsw.CT_from_rho(self.sig_bins + 1000,
                                        self.sig_mean_sa,
                                        self.p_ref)[0]
        mean_props = np.array([self.sig_mean_sa, self.sig_mean_ct])

        # interpoloator of mean properties from sigma
        self.sig_mean_ier = interp1d(self.sig_bins, mean_props,
                                     fill_value='extrapolate')

        self.sig_ref = np.mean(self.xy_sig[0, :])
        self.sa_ct_ref = self.sig_mean_ier(self.sig_ref)

        der = gsw.rho_first_derivatives(*self.sa_ct_ref, self.p_ref)
        self.beta_0, self.alpha_0, _  = der

        ct_diff = self.xsig_ct - self.sig_mean_ct[:, None]
        sa_diff = self.xsig_sa - self.sig_mean_sa[:, None]

        self.xsig_gamma = np.sign(ct_diff) \
                        * np.sqrt((self.alpha_0 * ct_diff) ** 2
                                  + (self.beta_0 * sa_diff) ** 2)

        # spice in xy space
        xy_mean = self.sig_mean_ier(self.xy_sig)
        self.xy_mean_sa = xy_mean[0]
        self.xy_mean_ct = xy_mean[1]

        ct_diff = self.xy_ct - self.xy_mean_ct
        sa_diff = self.xy_sa - self.xy_mean_sa


        self.xy_gamma = np.sign(ct_diff) \
                      * np.sqrt((self.alpha_0 * ct_diff) ** 2
                                + (self.beta_0 * sa_diff) ** 2)

    def sa_ct_from_sig_gamma(self, sig, gamma):
        """inverse of spice function"""
        if self.spice_def == 0:
            return SA_CT_from_sigma0_spiciness0(sig, gamma)
        elif self.spice_def == 2:
            return gamma, gsw.CT_from_rho(sig + 1e3, gamma, self.p_ref)[0]


        sig_err = 1e-4
        sig = np.asarray(sig)
        gamma = np.asarray(gamma)

        xy_mean = self.sig_mean_ier(sig)
        xy_pert = np.zeros_like(xy_mean)

        # start with estimate based on spice gradients
        dt_test = self._adjust_spice(gamma)
        xy_pert += dt_test

        curr_err = np.full_like(sig, sig_err + 1)  # always run one loop

        # run untill worst value below minimum error
        while np.any(np.abs(curr_err) > sig_err):
            # refine estimate of sigma
            sa_ct_est = xy_mean + xy_pert
            sig_est = gsw.rho(sa_ct_est[0], sa_ct_est[1], self.p_ref) - 1000
            d_sig = sig_est - sig
            dt_test = self._adjust_sigma(sa_ct_est, d_sig)
            xy_pert += dt_test

            # recompute spice
            g_est = np.sign(xy_pert[1]) * np.sqrt((self.alpha_0 * xy_pert[1]) ** 2
                                                 + (self.beta_0 * xy_pert[0]) ** 2)
            d_gamma = gamma - g_est
            dt_test = self._adjust_spice(d_gamma)
            xy_pert += dt_test

            # final estimate of sigma, gamma
            g_est = np.sign(xy_pert[1]) * np.sqrt((self.alpha_0 * xy_pert[1]) ** 2
                                                + (self.beta_0 * xy_pert[0]) ** 2)
            sa_ct_est = xy_mean + xy_pert
            sig_est = gsw.rho(*sa_ct_est, self.p_ref) - 1000
            curr_err = sig_est - sig

        return sa_ct_est[0], sa_ct_est[1]

    def _adjust_spice(self, d_gamma):
        """Adjust from current xy by d_gamma"""

        # linearized guess at properties
        theta = np.arctan2(-self.alpha_0, self.beta_0)
        d_ct = (d_gamma / np.abs(self.alpha_0) / np.sqrt(2))
        d_sa = (d_gamma / np.abs(self.alpha_0) / np.sqrt(2)) * np.tan(theta)

        return np.array([d_sa, d_ct])

    def _adjust_sigma(self, sa_ct, d_sig):
        """refine estimate of sigma"""
        beta, alpha, _  = gsw.rho_first_derivatives(*sa_ct, self.p_ref)

        theta = np.arctan2(-alpha, beta)
        d_ct = (d_sig / np.abs(alpha) / 2)
        d_sa = (d_sig / np.abs(alpha) / 2) * np.tan(-theta)

        return np.array([d_sa, d_ct])
