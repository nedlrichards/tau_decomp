"""Compare results of gridded and reconstructed total field"""

import numpy as np
from os.path import join
from scipy.interpolate import UnivariateSpline

from src import Section, sonic_layer_depth
from src import grid_field, SA_CT_from_sigma0_spiciness0, append_climatolgy
from src import RDModes, Config

import pyducts

sec4 = Section()
fc = 400
save_dir = f'data/processed/field_{int(fc)}'
if False:
    save_dir = join('/hb/scratch/edrichar/computed_results/', save_dir)

total_lvls  = np.load('data/processed/inputed_spice.npz')['lvls']
stab_spice = sec4.stable_spice(total_lvls)
stab_lvls = sec4.stable_cntr_height(stab_spice)

#stab_spice  = np.load('data/processed/inputed_spice.npz')['lvls']

z_a, c_bg = sec4.compute_c_field(stab_lvls)
z_a, c_tilt = sec4.compute_c_field(stab_spice)

sig_bg, tau_bg = grid_field(sec4.z_a, stab_lvls, sec4.sig_lvl)
sig_tilt, tau_tilt = grid_field(sec4.z_a, stab_spice, sec4.sig_lvl)

delta_spice = sec4.spice - tau_tilt
tau_spice = tau_bg + delta_spice

sa_spice, ct_spice = SA_CT_from_sigma0_spiciness0(sig_bg, tau_spice)

_, _, _, c_spice = append_climatolgy(sec4.z_a, ct_spice, sa_spice,
                                     sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

_, _, _, c_total = append_climatolgy(sec4.z_a, sec4.theta, sec4.salinity,
                                     sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

sld_z, _ = sonic_layer_depth(z_a, c_bg, z_max=150)
sld_m = z_a[:, None] > sld_z
c_sld = np.ma.array(c_bg, mask=sld_m)
mean_c = np.mean(c_sld, axis=0).data

def run_ram(rf, xs, x_a, cg):
    """ram is the model of choice"""

    # RD TL from ram
    rf.write_frontmatter()

    x_i = np.bitwise_and(x_a >= xs, x_a < xs + rmax)

    for x, c_p in zip(x_a[x_i], cg.T[x_i]):
        #c_ier = UnivariateSpline(z_a, c_p, s=1, k=1)
        #z_ds = c_ier.get_knots()
        #c_ds = c_ier.get_coeffs()
        #rf.write_profile(x - xs, z_ds, c_ds)
        rf.write_profile(x - xs, z_a, c_p)

    pyducts.ram.run_ram()

    zplot, rplot, p_ram = pyducts.ram.read_grid()
    return zplot, xs + rplot, p_ram, x_i

# split transect into small sections
rmax = 60e3
d_section = 10e3
x_start = np.arange(int((sec4.x_a[-1] - rmax) / d_section) + 1) * d_section
D = z_a[-1]
z_save = 150.  # restrict size of PE result

def save_tl(xs, fc, z_save, c_bg, c_tilt, c_spice, c_total, save_couple=True):
    cf = Config(fc)

    rf = pyducts.ram.RamIn(cf.fc, cf.z_src, rmax, D,
                           bottom_HS=cf.bottom_HS, dr=100., zmax_plot=D)

    tmp_dict = {"z_a":z_a}
    tmp_dict["xs"] = xs

    sec4 = Section()

    zplot, rplot, p_bg, x_i = run_ram(rf, xs, sec4.x_a, c_bg)
    tmp_dict["x_a"] = sec4.x_a[x_i]
    tmp_dict["zplot"] = zplot[zplot <= z_save]
    tmp_dict["rplot"] = rplot
    tmp_dict["c_bg"] = c_bg[:, x_i]
    tmp_dict["p_bg"] = p_bg[:, zplot <= z_save]
    tmp_dict["fc"] = cf.fc

    zplot, rplot, p_tilt, x_i = run_ram(rf, xs, sec4.x_a, c_tilt)
    tmp_dict["c_tilt"] = c_tilt[:, x_i]
    tmp_dict["p_tilt"] = p_tilt[:, zplot <= z_save]

    zplot, rplot, p_spice, x_i = run_ram(rf, xs, sec4.x_a, c_spice)
    tmp_dict["c_spice"] = c_spice[:, x_i]
    tmp_dict["p_spice"] = p_spice[:, zplot <= z_save]

    zplot, rplot, p_total, x_i = run_ram(rf, xs, sec4.x_a, c_total)
    tmp_dict["c_total"] = c_total[:, x_i]
    tmp_dict["p_total"] = p_total[:, zplot <= z_save]


    if save_couple:
        rd_modes = RDModes(tmp_dict['c_bg'], tmp_dict['x_a'], tmp_dict['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds, s=None)

        tmp_dict['r_modes'] = (rd_modes.r_plot + tmp_dict['xs']) / 1e3
        tmp_dict['bg_mode_amps'] = rd_modes.couple_cn()
        tmp_dict['psi_bg'] = rd_modes.psi_bg
        tmp_dict['k_bg'] = rd_modes.k_bg
        #psi_k_bg = (rd_modes.psi_bg, rd_modes.k_bg)
        print('bg')

        rd_modes = RDModes(tmp_dict['c_tilt'], tmp_dict['x_a'], tmp_dict['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds,)
                        #psi_k_bg=psi_k_bg)
        tmp_dict['tilt_mode_amps'] = rd_modes.couple_cn()
        print('tilt')

        rd_modes = RDModes(tmp_dict['c_spice'], tmp_dict['x_a'], tmp_dict['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds,)
                        #psi_k_bg=psi_k_bg)
        tmp_dict['spice_mode_amps'] = rd_modes.couple_cn()
        print('spice')

        rd_modes = RDModes(tmp_dict['c_total'], tmp_dict['x_a'], tmp_dict['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds,)
                        #psi_k_bg=psi_k_bg)
        tmp_dict['total_mode_amps'] = rd_modes.couple_cn()

    np.savez(join(save_dir, f'tl_section_{int(xs/1e3):03d}'), **tmp_dict)
    print(f'saved tl_section_{int(xs/1e3)}')

run_func = lambda xs: save_tl(xs, fc, z_save, c_bg, c_tilt, c_spice, c_total, save_couple=True)
run_func(690e3)
1/0
#list(map(run_func, x_start))
#run_func(420e3)
