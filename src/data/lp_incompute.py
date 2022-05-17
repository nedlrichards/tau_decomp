"""Incompute spice assuming spice change follows lower, complete, isopycnals"""

import numpy as np
from src import SectionLvls
from src import grid_field, append_climatolgy

def patch_spice(patch_index, filled_spice):
    """patch up spice series from denser isopycnals"""
    full_i = patch_index + 1
    # patch by integrating difference of complete timeseries
    tau_diff = np.diff(filled_spice[1, full_i, :])

    nan_i = np.isnan(filled_spice[1, patch_index, :])

    patched_lvl = np.copy(filled_spice[1, patch_index, :])

    # list of nan_indices
    sections = []
    sec_start = np.argmax(nan_i)
    while nan_i[sec_start] and sec_start < nan_i.size - 1:
        if np.all(nan_i[sec_start:]):
            sec_end = nan_i.size - 1
        else:
            sec_end = np.argmax(~nan_i[sec_start:]) + sec_start
        sections.append([sec_start, sec_end])
        sec_start = np.argmax(nan_i[sec_end: ]) + sec_end

    for i, sec in enumerate(sections):
        if sec[0] != 0:
            f = tau_diff[sec[0] - 1: sec[1] - 1]
            fill_r = patched_lvl[sec[0] - 1] + np.cumsum(f)
        else:
            fill_r = np.full(sec[1] - sec[0], np.nan)

        if sec[1] != nan_i.size - 1:
            f = tau_diff[sec[0]: sec[1]]
            fill_l = patched_lvl[sec[1]] - np.cumsum(f[::-1])
            fill_l = fill_l[::-1]
        else:
            fill_l = np.full(sec[1] - sec[0], np.nan)
        all_fill = np.nanmean(np.array([fill_r, fill_l]), axis=0)

        patched_lvl[sec[0]:sec[1]] = all_fill

    return patched_lvl

sec4 = SectionLvls()
start_spice = sec4.lvls.copy()
filled_spice = sec4.lvls.copy()

last_full_i = np.argmax(~np.any(np.isnan(start_spice[1, :, :]), axis=1))
all_nan_i = np.argmax(~np.all(np.isnan(start_spice[1, :, :]), axis=1))

for i in range(last_full_i, all_nan_i, -1):
    patched_lvl = patch_spice(i, filled_spice)
    filled_spice[1, i, :] = patched_lvl

stable_spice = sec4.stable_spice(filled_spice)
stable_lvls = sec4.stable_cntr_height(filled_spice)
stable_spice_lvls = sec4.stable_cntr_height(stable_spice)

z_a, c_bg = sec4.compute_c_field(stable_spice_lvls)
z_a, c_tilt = sec4.compute_c_field(stable_spice)
z_a, c_spice_dmr = sec4.compute_c_field(stable_lvls)

sig_bg, tau_bg = grid_field(sec4.z_a, stable_spice_lvls, sec4.sig_lvl)
sig_tilt, tau_tilt = grid_field(sec4.z_a, stable_spice, sec4.sig_lvl)

delta_spice = sec4.spice - tau_tilt
tau_spice = tau_bg + delta_spice

sa_spice, ct_spice = sec4.field.sa_ct_from_sig_gamma(sig_bg, tau_spice)

_, _, _, c_spice = append_climatolgy(sec4.z_a, ct_spice, sa_spice,
                                        sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

_, _, _, c_total = append_climatolgy(sec4.z_a, sec4.theta, sec4.salinity,
                                        sec4.z_clim, sec4.temp_clim, sec4.sal_clim)

save_dict = {'filled_lvls':filled_spice, 'stable_spice':stable_spice,
                'stable_spice_lvls':stable_spice_lvls,
                'stable_lvls':stable_lvls, 'c_bg':c_bg, 'c_tilt':c_tilt,
                'c_spice':c_spice, 'c_spice_dmr':c_spice_dmr,
                'c_total':c_total, 'sig':sec4.sig_lvl,
                'x_a':sec4.x_a, 'z_a':z_a}

np.savez('data/processed/inputed_decomp.npz', **save_dict)
