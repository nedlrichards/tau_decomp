import numpy as np
from scipy.io import loadmat
import gsw

from src import Config

class LocalSpice:
    """Generate spice, gamma, in local coordinates"""

    def __init__(self, num_sigma=300):
        """Standard processing"""
        self.cf = Config()
        grid_data = loadmat('data/raw/section4_fields.mat')
        self.num_sigma = num_sigma

        self.x_a = np.squeeze(grid_data['xx'])
        self.z_a = np.squeeze(grid_data['zz'])

        self.press = gsw.p_from_z(-self.z_a, self.cf.lat)

        self.xy_sa = gsw.SA_from_SP(grid_data['sali'], self.press[:, None],
                                 self.cf.lon, self.cf.lat)
        self.xy_ct = grid_data['thetai']
        self.xy_sig = gsw.sigma0(self.xy_sa, self.xy_ct)
        self.r_prof = np.broadcast_to(x_a, self.xy_sa.shape)

        # derivatives at local origin of spice

        self.alpha  = np.nanmean(gsw.alpha(self.xy_sa,
                                              self.xy_ct,
                                              self.press[:, None]))
        self.beta = np.nanmean(gsw.beta(self.xy_sa,
                                             self.xy_ct,
                                             self.press[:, None]))

        # grid salinity and theta into density bins
        self.sig_bins = np.linspace(self.xy_sig.min(),
                                    self.xy_sig.max(),
                                    num=self.num_bins)[1:]

        prof_iter = zip(self.xy_sig.T, self.xy_sa.T, self.xy_theta.T)

        props_xsig = []
        for sig_prof, sa_prof, ct_prof in prof_iter:
            i_sa = np.interp(bins, sig_prof, sa_prof, left=np.nan, right=np.nan)
            i_ct = np.interp(bins, sig_prof, ct_prof, left=np.nan, right=np.nan)
            props_xsig.append(np.array([i_sa, i_ct]))
        props_xsig = np.stack(props_xsig, axis=2)
        self.xsig_sa = props[0, :, :]
        self.xsig_ct = props[1, :, :]

        mean_props = np.nanmean(props_xsig, axis=-1)
        self.sig_mean_sa = mean_props[0]
        self.sig_mean_ct = mean_props[1]

        ct_diff = self.xsig_ct -self.sig_mean_ct
        sa_diff = self.xsig_sa - self.sig_mean_sa

        self.xsig_spice = np.sign(ct_diff) \
                        * np.sqrt((self.alpha * 1e3 * ct_diff) ** 2
                                  + (self.beta * 1e3 * sa_diff) ** 2)

        # spice in xy space
        f_sig = self.xy_sig.flatten()
        mean_sa_y = np.interp(f_sig, self.sig_bins, self.sig_mean_sa)
        mean_sa_y = mean_sa_y.reshape(self.xy_sig.shape)
        mean_ct_y = np.interp(f_sig, self.sig_bins, self.sig_mean_ct)
        mean_ct_y = mean_ct_y.reshape(self.xy_sig.shape)

        ct_diff = self.xy_ct - mean_ct_y
        sa_diff = self.xy_sa - mean_sa_y

        self.xy_spice = np.sign(ct_diff) \
                      * np.sqrt((self.alpha * 1e3 * ct_diff) ** 2
                                + (self.beta * 1e3 * sa_diff) ** 2)
