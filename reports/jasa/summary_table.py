import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

import matplotlib as mpl
custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}", # for the align enivironment
    }

mpl.rcParams.update(custom_preamble)

plt.style.use('elr')
plt.ion()

#source_depth="shallow"
source_depth="deep"
range_bounds = np.array((7.5e3, 47.5e3))
ml_tl_rb = (7.5e3, 37.0e3)
deep_rb = (7.5e3, 40.0e3)
savedir = 'reports/jasa/tex'
cf = Config()

load_dir = 'data/processed/'

shallow_int_400 = np.load(join(load_dir, 'int_eng_shallow_400.npz'))
shallow_int_1000 = np.load(join(load_dir, 'int_eng_shallow_1000.npz'))

r_a = shallow_int_400['r_a']
block_i = shallow_int_400['block_i'] & shallow_int_1000['block_i']

deep_int_400 = np.load(join(load_dir, 'int_eng_deep_400.npz'))
deep_int_1000 = np.load(join(load_dir, 'int_eng_deep_1000.npz'))

def z_nn(v):
    """Format negative number close to zero as 0.0"""
    return 0.0 if round(v,1) == 0 else v

def one_stat(stats, range_bounds):

    lin_rgs = stats["mean_rgs"]
    mean = lin_rgs.intercept + range_bounds * lin_rgs.slope
    lin_rgs = stats['rms_rgs']
    rms = lin_rgs.intercept + range_bounds * lin_rgs.slope

    left_str = r"{"
    left_str += rf"\textbf{{{z_nn(mean[0]): 3.1f}}}"
    left_str += r" \\"
    left_str += rf" \quad $\pm$ {z_nn(rms[0]):3.1f}"
    left_str += "}"

    #left_str = rf"\textbf{{{z_nn(mean[0]): 3.1f}}}"
    #left_str += rf" $\pm$ {z_nn(rms[0]):3.1f}"

    rght_str = r"{"
    rght_str += rf" \textbf{{{z_nn(mean[-1]): 3.1f}}}"
    rght_str += r" \\"
    rght_str += rf" \quad $\pm$ {z_nn(rms[-1]):3.1f}"
    rght_str += "}"

    return left_str, rght_str

#demean_1000 = lines_1000[1:] - lines_1000[0]
#stats_1000 = [field_stats(r_a, v, range_bounds=range_bounds) for v in demean_1000]


out_str = ""

out_str += r"\begin{tblr}{|Q[l,t]|Q[l,t]|Q[l,t]|Q[l,t]|Q[l,t]|Q[l,t]|Q[l,t]|Q[l,t]|}" + " \n"
#out_str += r"\begin{tblr}{stretch=0,columns={5em,c,colsep=2pt},rows={1em,c,rowsep=3pt},} " + " \n"
out_str += r"7.5 km & 45 km & 7.5 km & 45 km &7.5 km & 45 km & 7.5 km & 45 km \\ " + " \n"
out_str += r"\hline " + " \n"

lines = shallow_int_400['ml_ml']
demean_400 = lines[1:] - lines[0]
stats_400 = [field_stats(r_a, v, range_bounds=range_bounds) for v in demean_400]

mode_lines = shallow_int_400['ml_proj']
demean_400_modes = mode_lines[1:] - mode_lines[0]
stats_400_modes = [field_stats(r_a, v, range_bounds=range_bounds) for v in demean_400_modes]

bg_i = cf.field_types.index('bg')
tilt_i = cf.field_types.index('tilt')
spice_i = cf.field_types.index('spice')
total_i = cf.field_types.index('total')

left_str, rght_str = one_stat(stats_400[bg_i], range_bounds)
out_str += left_str + " & - &" +  rght_str + " & -"

left_str, rght_str = one_stat(stats_400_modes[bg_i], range_bounds)
out_str += "& " + left_str + " & - &" +  rght_str + " & -" + " \\\\ \n"

"""
left_str, rght_str = one_stat(stats_400_modes[bg_i], range_bounds)
out_str += " & " + left_str + rght_str + " & - & - \\\\ \n"
"""

def one_field(inds):

    out_str = ""
    left_str, rght_str = one_stat(stats_400[inds], range_bounds)

    flt_eng = demean_400[inds][block_i[inds], :].copy()
    flt_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)
    ls_wo, rs_wo = one_stat(flt_stats, range_bounds)

    out_str += left_str + " & " + ls_wo  + " & " + rght_str + " & " + rs_wo

    left_str, rght_str = one_stat(stats_400_modes[inds], range_bounds)

    flt_eng = demean_400_modes[inds][block_i[inds], :].copy()
    flt_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)
    ls_wo, rs_wo = one_stat(flt_stats, range_bounds)

    out_str += " &" + left_str + " & " + ls_wo  + " & " + rght_str + " & " + rs_wo + " \\\\ \n"

    return out_str

out_str += one_field(tilt_i)
out_str += one_field(spice_i)
out_str += one_field(total_i)

"""
flt_eng = demean_400[spice_i][block_i[spice_i], :].copy()
flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)
ls_wo, rs_wo = one_stat(flt_spice_stats, range_bounds)
out_str += left_str + rght_str + " & " + ls_wo + rs_wo + " \\\\ \n"

left_str, rght_str = one_stat(stats_400[total_i], range_bounds)

flt_eng = demean_400[total_i][block_i[total_i], :].copy()
flt_spice_stats = field_stats(r_a, flt_eng, range_bounds=range_bounds)
ls_wo, rs_wo = one_stat(flt_spice_stats, range_bounds)
out_str += left_str + rght_str + " & " + ls_wo + rs_wo + " \\\\ \n"
"""

out_str += "\end{tblr}"

with open("./reports/jasa/tex/table.txt", "w") as f:
    f.write(out_str)

print(out_str)
1/0

# use a 2nd order fit of the mean for fit

rb_i = (r_a >= ml_tl_rb[0]) & (r_a <= ml_tl_rb[1])

test_400 = shallow_int_400['ml_tl'].copy()
bg = test_400[0]
bg_fit_400 = []
for line in bg:
    fit = np.polynomial.polynomial.polyfit(r_a[rb_i], line[rb_i], 1)
    bg_fit_400.append(np.polynomial.polynomial.polyval(r_a, fit))
bg_fit_400 = np.array(bg_fit_400)
test_400[0] = bg_fit_400

shallow_modelled = np.load(join(load_dir, 'tl_eng_model_400.npz'))['ml_pred']

test_1000 = shallow_int_1000['ml_tl'].copy()
bg = test_1000[0]
bg_fit_1000 = []
for line in bg:
    fit = np.polynomial.polynomial.polyfit(r_a[rb_i], line[rb_i], 2)
    bg_fit_1000.append(np.polynomial.polynomial.polyval(r_a, fit))
bg_fit_1000 = np.array(bg_fit_1000)
test_1000[0] = bg_fit_1000

shallow_modelled = np.load(join(load_dir, 'tl_eng_model_1000.npz'))['ml_pred']

fig, axes = plot_sparks(r_a, test_400, test_1000, ylim=(-20, 25), stat_range=ml_tl_rb)
fig.savefig(join(savedir, f'figure_17.pdf'), dpi=300)
