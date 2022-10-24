import numpy as np
from math import pi
from scipy.io import loadmat
from scipy.interpolate import interp1d
import gsw

from src import Config, SA_CT_from_sigma0_spiciness0

class Field:
    """Generate spice, gamma, in local coordinates"""

    def __init__(self, num_sigma=300):
        """Standard processing"""
        self.cf = Config()
        grid_data = loadmat('data/processed/stablized_field.mat')
        self.num_sigma = num_sigma

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

        self.xy_gamma = gsw.spiciness0(self.xy_sa, self.xy_ct)

    def sa_ct_from_sig_gamma(self, sig, gamma, sig_err=1e-4):
        return SA_CT_from_sigma0_spiciness0(sig, gamma)
