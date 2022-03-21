"""Incompute spice assuming spice change follows lower, complete, isopycnals"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from scipy.stats import linregress
from scipy.signal import filtfilt

from src import Section
from src import join_spice

plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')

sec4 = Section()
filled_spice = sec4.lvls.copy()
stab_spice = sec4.stable_spice(sec4.lvls)

last_full_i = np.argmax(~np.any(np.isnan(stab_spice[1, :, :]), axis=1))
all_nan_i = np.argmax(~np.all(np.isnan(stab_spice[1, :, :]), axis=1))

def patch_spice(patch_index):
    """patch up spice series from denser isopycnals"""
    full_i = patch_index + 1
    # patch by integrating difference of complete timeseries
    tau_diff = np.diff(filled_spice[1, full_i, :])

    nan_i = np.isnan(filled_spice[1, patch_index, :])


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

    for sec in sections:
        if sec[0] != 0:
            f = tau_diff[sec[0] - 1: sec[1] - 1]
            fill_r = filled_spice[1, patch_index, sec[0] - 1] \
                + np.cumsum(f)
        else:
            fill_r = np.full(sec[1] - sec[0], np.nan)

        if sec[1] != nan_i.size - 1:
            f = tau_diff[sec[0]: sec[1]]
            fill_l = filled_spice[1, patch_index, sec[1]] \
                    - np.cumsum(f[::-1])
            fill_l = fill_l[::-1]
        else:
            fill_l = np.full(sec[1] - sec[0], np.nan)
        all_fill = np.nanmean(np.array([fill_r, fill_l]), axis=0)
        all_fill = fill_l

        filled_spice[1, patch_index, sec[0]:sec[1]] = all_fill

for i in range(last_full_i, all_nan_i, -1):
    patch_spice(i)

save_dict = {'lvls':filled_spice, 'x_a':sec4.x_a, 'z_a':sec4.z_a}

np.savez('data/processed/inputed_spice.npz', **save_dict)
