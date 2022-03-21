import numpy as np
from scipy.io import loadmat
from os.path import join
from scipy.interpolate import UnivariateSpline

from src import Section
from src import grid_field, SA_CT_from_sigma0_spiciness0, append_climatolgy


sec4 = Section()

imputed_spice  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_spice = sec4.stable_spice(imputed_spice)
stab_lvls = sec4.stable_cntr_height(stab_spice)

sig_bg, tau_bg = grid_field(sec4.z_a, stab_lvls, sec4.sig_lvl)
sig_tilt, tau_tilt = grid_field(sec4.z_a, stab_spice, sec4.sig_lvl)

delta_spice = sec4.spice - tau_tilt
tau_spice = tau_bg + delta_spice

sa_spice, ct_spice = SA_CT_from_sigma0_spiciness0(sig_bg, tau_spice)

_, _, _, c_spice = append_climatolgy(sec4.z_a, ct_spice, sa_spice,
                                     sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

_, _, _, c_total = append_climatolgy(sec4.z_a, sec4.theta, sec4.salinity,
                                     sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

z_a, c_bg = sec4.compute_c_field(stab_lvls)
z_a, c_tilt = sec4.compute_c_field(stab_spice)

save_dir = {'x_a':sec4.x_a, 'z_a':z_a, 'c_bg':c_bg, 'c_spice':c_spice,
            'c_tilt':c_tilt, 'c_total':c_total}
np.savez('data/processed/decomposed_fields.npz', **save_dir)
