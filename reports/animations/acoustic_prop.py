"""Compare results of gridded and reconstructed total field"""

import numpy as np
from math import pi
from os.path import join
from scipy.interpolate import UnivariateSpline

import argparse

from src import RDModes, Config, section_cfield

import pyducts

parser = argparse.ArgumentParser(description='Transmission simulations')
parser.add_argument('fc', metavar='F', type=float,
                    help='Frequency of simulations')
parser.add_argument('source_depth', metavar='S', type=str,
                    help='deep or shallow')
parser.add_argument('-d_section', dest='d_section', type=int,
                    default=10, help='distance between sections')

args = parser.parse_args()

fc = args.fc
source_depth = args.source_depth
d_section = args.d_section * 1e3
d_max = 300e3

save_dir = f'reports/animations//processed/field_{int(fc)}_'+ source_depth

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
x_start = np.arange(np.ceil(d_max / 1e3)) * 1e3
D = z_a[-1]
z_save = 200.  # restrict size of PE result

def save_tl(xs, z_save, save_couple=True):

    rf = pyducts.ram.RamIn(cf.fc, cf.z_src, cf.rmax, D,
                           bottom_HS=cf.bottom_HS, dr=100., zmax_plot=D)

    tmp_dict = {"z_a":z_a, "xs":xs, "fc":cf.fc, "z_src":cf.z_src}

    x_sec, c_total_sec = section_cfield(xs, x_a, c_total, rmax=cf.rmax)
    zplot, rplot, p_bg = run_ram(rf, x_sec, z_a, c_total_sec)
    tmp_dict["x_a"] = x_sec
    tmp_dict["zplot"] = zplot[zplot <= z_save]
    tmp_dict["rplot"] = rplot
    tmp_dict["p_bg"] = p_bg[:, zplot <= z_save]

    np.savez(join(save_dir, f'tl_section_{int(xs/1e3):03d}'), **tmp_dict)
    print(f'saved tl_section_{int(xs/1e3)}')

run_func = lambda xs: save_tl(xs, z_save, save_couple=False)

list(map(run_func, x_start))
