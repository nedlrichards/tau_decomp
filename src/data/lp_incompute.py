"""Compare results of gridded and reconstructed total field"""

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

stab_spice_patch = sec4.stable_spice(filled_spice)

fig, ax = plt.subplots()

for i in range(24, 18, -1):
    ax.plot(sec4.x_a / 1e3, filled_spice[1, i, :], 'C1')
    ax.plot(sec4.x_a / 1e3, sec4.lvls[1,  i, :], 'k')

ax.set_xlim([500, 970])
ax.set_ylim([1.5, 2.5])

fig, ax = plt.subplots()

for i in range(24, 18, -1):
    ax.plot(sec4.x_a / 1e3, filled_spice[1, i, :], 'k')
    ax.plot(sec4.x_a / 1e3, stab_spice_patch[1,  i, :], 'C0')

ax.set_xlim([500, 970])
ax.set_ylim([1.5, 2.5])

1/0

fill_val = np.mean([tau_diff[section[0]-1], tau_diff[section[1]+1]])

filled_spice[1, patch_i, section[0]: section[1]] = fill_val \
        + filled_spice[1, last_full_i, section[0]: section[1]]

last_full_i -= 1

patch_i = last_full_i - 1
tau_diff = filled_spice[1, patch_i, :] - filled_spice[1, last_full_i, :]

nan_i = np.isnan(filled_spice[1, patch_i, :])

# list of nan_indices
section = np.argmax(nan_i)
section = [section, np.argmax(~nan_i[section:]) + section]

fill_val = np.mean([tau_diff[section[0]-1], tau_diff[section[1]+1]])

filled_spice[1, patch_i, section[0]: section[1]] = fill_val \
        + filled_spice[1, last_full_i, section[0]: section[1]]

section = np.argmax(nan_i[section[1]:]) + section[1]
section = [section, np.argmax(~nan_i[section:]) + section]

fill_val = np.mean([tau_diff[section[0]-1], tau_diff[section[1]+1]])

filled_spice[1, patch_i, section[0]: section[1]] = fill_val \
        + filled_spice[1, last_full_i, section[0]: section[1]]


fig, ax = plt.subplots()
ax.plot(sec4.x_a / 1e3, filled_spice[1, last_full_i, :], 'k')

ax.plot(sec4.x_a / 1e3, filled_spice[1, patch_i, :], 'k')
ax.plot(sec4.x_a / 1e3, sec4.lvls[1, patch_i, :], 'C1')

ax.set_xlim([0, 130])
ax.set_ylim([1.5, 2.1])


1/0

fig, ax = plt.subplots()
ax.plot(sec4.x_a / 1e3, sec4.lvls[1, last_full_i, :], 'C0')
ax.plot(sec4.x_a / 1e3, sec4.lvls[1, patch_i, :], 'C1')

patch_i = last_full_i - 1
filled_spice[:, patch_i, nan_i] = filled_spice[:, last_full_i, nan_i]

ax.plot(sec4.x_a / 1e3, filled_spice[1, patch_i, :], 'k')

filt_spice = filtfilt(sec4.b_lp, sec4.a_lp,
        filled_spice[1, patch_i, :], method="gust")
filled_spice[1, patch_i, :] = filt_spice

ax.plot(sec4.x_a / 1e3, filled_spice[1, last_full_i, :], 'C0')
ax.plot(sec4.x_a / 1e3, filled_spice[1, patch_i, :], 'C1')

ax.set_xlim([0, 130])
ax.set_ylim([1.5, 2.1])


1/0

patch_i = last_full_i - 1

# choose the section to fit over

nan_i = np.isnan(filled_spice[1, patch_i, :])
last_nan_i = nan_i.size - np.argmax(nan_i[::-1]) - 1

last_lr = linregress(sec4.x_a[:last_nan_i], sec4.lvls[1, last_full_i, :last_nan_i])
curr_lr = linregress(sec4.x_a[:last_nan_i], sec4.lvls[1, patch_i, :last_nan_i])

fig, ax = plt.subplots()
ax.plot(sec4.x_a / 1e3, sec4.lvls[1, last_full_i, :], 'k')
ax.plot(sec4.x_a / 1e3, sec4.lvls[1, patch_i, :], 'k')

ax.plot(sec4.x_a / 1e3, last_lr.intercept + sec4.x_a * last_lr.slope, 'C0')
ax.plot(sec4.x_a / 1e3, curr_lr.intercept + sec4.x_a * curr_lr.slope, 'C0')

ax.set_xlim([0, 130])
ax.set_ylim([1.5, 2.1])

1/0



fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 4))
for i in cntr_indx:
    axes[0].plot(sec4.x_a / 1e3, sec4.lvls[0, i, :].T, 'k')

    axes[1].plot(sec4.x_a / 1e3, sec4.lvls[1, i, :].T, 'k')
    axes[1].plot(sec4.x_a / 1e3, stab_spice[1, i, :].T, 'C1')

axes[1].set_xlabel('Range (km)')
axes[0].set_ylabel('Depth (m)')
axes[1].set_ylabel(r'$\tau$ (kg/m$^3$)')
axes[0].set_xlim([0, 130])
axes[0].set_ylim([110, 10])
axes[1].set_ylim([1.5, 2.1])

pos = axes[0].get_position()
pos.x0 -= 0.02
pos.x1 += 0.08
pos.y0 += 0.03
pos.y1 += 0.08
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 -= 0.02
pos.x1 += 0.08
pos.y0 += 0.00
pos.y1 += 0.05
axes[1].set_position(pos)

fig.savefig('figures/stable_properties.png')
