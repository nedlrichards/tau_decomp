"""Extrapolate, lowpass, truncate and otherwise manipulate contours"""

import numpy as np
from scipy.signal import iirfilter, sosfiltfilt, sosfreqz, filtfilt, freqz

def not_nan_segments(cntr):
    """generator that return indicies of segment of defined values"""

    nan_i = np.isnan(cntr[0])
    if np.all(nan_i):
        # no defined values
        return (0, 0)

    not_nan = 0
    last_nan = 0

    while not_nan < cntr.shape[-1] - 1:
        # fill segments with spice values from denser segment
        last_nan = not_nan

        # move to next segment of defined values
        not_nan += np.argmax(~nan_i[not_nan:])

        # define the end of the defined values
        if np.any(nan_i[not_nan:]):
            next_nan = np.argmax(nan_i[not_nan:])
        else:
            next_nan = cntr.shape[-1] - not_nan


        if next_nan == 0:
            # check for all nans at the end of contour
            return (0, 0)
        else:
            yield (not_nan, not_nan + next_nan)

        # reset loop condition
        not_nan = not_nan + next_nan


def join_spice(lvls):
    """Extrapolate spice between profiles by using denser contour value"""
    extrp_spice_lvls = lvls.copy()

    # iterate starting at furthest range, and densest contour
    lvl_iter = np.moveaxis(extrp_spice_lvls, 1, 0)[:: -1, :, :: -1]

    for lvl, last_lvl in zip(lvl_iter[1:], lvl_iter[:-1]):
        last_nan = 0
        all_inds = []
        for inds in not_nan_segments(lvl):
            all_inds.append(inds)
            if inds[-1] == 0:
                last_nan = inds[-1]
                continue

            if last_nan == 0:
                if inds[0] > 0:
                    lvl[1, :inds[0]] = last_lvl[1, :inds[0]]
                last_nan = inds[-1]
                continue
            lvl[1, last_nan: inds[0]] = last_lvl[1, last_nan: inds[0]]
            last_nan = inds[-1]
        if last_nan < lvl.shape[-1]:
            lvl[1, last_nan:] = last_lvl[1, last_nan:]

    return extrp_spice_lvls

def reduce_field(basic_lvls, lp_lvls):
    """Total field that can be represented by decompostion"""

    reduce_lvls = basic_lvls.copy()

    # iterate starting at furthest range, and densest contour
    red_lvl_iter = np.moveaxis(reduce_lvls, 1, 0)[:: -1, :, :: -1]
    lp_lvl_iter = np.moveaxis(lp_lvls, 1, 0)[:: -1, :, :: -1]

    # set all spice values to lp value, except for the first segment
    for red_lvl, lp_lvl in zip(red_lvl_iter, lp_lvl_iter):
        inds = list((not_nan_segments(red_lvl)))
        if not inds:
            # indicates all nans
            continue
        ind = inds[0]
        red_lvl[1, ind[-1]:] = lp_lvl[1, ind[-1]:]

    return reduce_lvls
