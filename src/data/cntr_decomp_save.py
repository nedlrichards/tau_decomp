"""Compare results of gridded and reconstructed total field"""

import numpy as np
from math import pi
from os.path import join
from scipy.interpolate import UnivariateSpline

import argparse

from src import SectionLvls, RDModes, Config, section_cfield

import pyducts

parser = argparse.ArgumentParser(description='Transmission simulations')
parser.add_argument('fc', metavar='F', type=float,
                    help='Frequency of simulations')
parser.add_argument('source_depth', metavar='S', type=str,
                    help='deep or shallow')
parser.add_argument('-d_section', dest='d_section', type=int,
                    default=10, help='distance between sections')

args = parser.parse_args()

sec4 = SectionLvls()
fc = args.fc
source_depth = args.source_depth
d_section = args.d_section * 1e3

save_dir = f'data/processed/field_{int(fc)}_'+ source_depth
if False:
    save_dir = join('/hb/scratch/edrichar/computed_results/', save_dir)

cf = Config(source_depth=source_depth, fc=fc)
fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_spice = fields['c_spice']
c_tilt = fields['c_tilt']
c_total = fields['c_total']

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

# split transect into small sections
x_start = np.arange(int((sec4.x_a[-1] - cf.rmax) / d_section) + 1) * d_section
D = z_a[-1]
z_save = 150.  # restrict size of PE result

def compute_rd_modes(c_field, x_a, z_a, cf, psi_k=None):
        rd_modes = RDModes(c_field, x_a, z_a, cf, psi_k_bg=psi_k)

        ll = -2 * pi / (np.diff(rd_modes.k_bg))
        p_i = np.argmax(ll)
        m_range = (-50, 170)

        cm_i = np.arange(p_i + m_range[0], p_i + m_range[1])
        cm_i = cm_i[cm_i >= 0]

        rd_trunc = RDModes(c_field, x_a, z_a, cf,
                           psi_k_bg=(rd_modes.psi_bg[cm_i, :],
                                     rd_modes.k_bg[cm_i]))

        trunc_mode_amps = rd_trunc.couple_cn()
        r_modes = (rd_modes.r_plot + x_a[0]) / 1e3

        return r_modes, trunc_mode_amps, rd_trunc.psi_bg, rd_trunc.k_bg


def save_tl(xs, z_save, save_couple=True):

    rf = pyducts.ram.RamIn(cf.fc, cf.z_src, cf.rmax, D,
                           bottom_HS=cf.bottom_HS, dr=100., zmax_plot=D)

    tmp_dict = {"z_a":z_a, "xs":xs, "fc":cf.fc, "z_src":cf.z_src}

    x_sec, c_bg_sec = section_cfield(xs, x_a, c_bg, rmax=cf.rmax)
    zplot, rplot, p_bg = run_ram(rf, x_sec, z_a, c_bg_sec)
    tmp_dict["x_a"] = x_sec
    tmp_dict["zplot"] = zplot[zplot <= z_save]
    tmp_dict["rplot"] = rplot
    tmp_dict["p_bg"] = p_bg[:, zplot <= z_save]

    x_sec, c_tilt_sec = section_cfield(xs, x_a, c_tilt, rmax=cf.rmax)
    _, _, p_tilt = run_ram(rf, x_sec, z_a, c_tilt_sec)
    tmp_dict["p_tilt"] = p_tilt[:, zplot <= z_save]

    x_sec, c_spice_sec = section_cfield(xs, x_a, c_spice, rmax=cf.rmax)
    _, _, p_spice = run_ram(rf, x_sec, z_a, c_spice_sec)
    tmp_dict["p_spice"] = p_spice[:, zplot <= z_save]

    x_sec, c_total_sec = section_cfield(xs, x_a, c_total, rmax=cf.rmax)
    _, _, p_total = run_ram(rf, x_sec, z_a, c_total_sec)
    tmp_dict["p_total"] = p_total[:, zplot <= z_save]


    if save_couple:
        x_sec, c_bg_sec = section_cfield(xs, x_a, c_bg, rmax=cf.rmax)
        out = compute_rd_modes(c_bg_sec, x_sec, z_a, cf)

        tmp_dict['r_modes'] = out[0]
        tmp_dict['bg_mode_amps'] = out[1]
        tmp_dict['psi_bg'] = out[2]
        tmp_dict['k_bg'] = out[3]
        print('bg')

        x_sec, c_tilt_sec = section_cfield(xs, x_a, c_tilt, rmax=cf.rmax)
        out = compute_rd_modes(c_tilt_sec, x_sec, z_a, cf)
        tmp_dict['tilt_mode_amps'] = out[1]
        tmp_dict['psi_tilt'] = out[2]
        tmp_dict['k_tilt'] = out[3]
        print('tilt')

        x_sec, c_spice_sec = section_cfield(xs, x_a, c_spice, rmax=cf.rmax)
        out = compute_rd_modes(c_spice_sec, x_sec, z_a, cf)
        tmp_dict['spice_mode_amps'] = out[1]
        tmp_dict['psi_spice'] = out[2]
        tmp_dict['k_spice'] = out[3]
        print('spice')

        x_sec, c_total_sec = section_cfield(xs, x_a, c_total, rmax=cf.rmax)
        out = compute_rd_modes(c_total_sec, x_sec, z_a, cf)
        tmp_dict['total_mode_amps'] = out[1]
        tmp_dict['psi_total'] = out[2]
        tmp_dict['k_total'] = out[3]
        print('total')
    else:
        x_sec, c_bg_sec = section_cfield(xs, x_a, c_bg, rmax=cf.rmax)
        rd_modes = RDModes(c_bg_sec, x_a, z_a, cf)

        ll = -2 * pi / (np.diff(rd_modes.k_bg))
        p_i = np.argmax(ll)
        m_range = (-50, 170)

        cm_i = np.arange(p_i + m_range[0], p_i + m_range[1])
        cm_i = cm_i[cm_i >= 0]

        tmp_dict['psi_bg'] = rd_modes.psi_bg[cm_i, :]
        tmp_dict['k_bg'] = rd_modes.k_bg[cm_i]


    np.savez(join(save_dir, f'tl_section_{int(xs/1e3):03d}'), **tmp_dict)
    print(f'saved tl_section_{int(xs/1e3)}')

run_func = lambda xs: save_tl(xs, z_save, save_couple=False)

list(map(run_func, x_start))
