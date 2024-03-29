"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import gsw
from scipy.signal import iirfilter, filtfilt
from scipy.stats import linregress

from src import lvl_profiles, grid_field
from src import append_climatolgy
from src import Config
from src import Field

climatology_file = 'data/raw/NPAL05prosGDEM.mat'

class SectionLvls:
    """Common data used in section processing"""

    def __init__(self, spice_def=0):
        """Load data and compute commonly used quanitities"""

        self.cf = Config()
        self.spice_def = spice_def
        self.field = Field(spice_def=spice_def)

        self.x_a = self.field.x_a
        self.dx = (self.x_a[-1] - self.x_a[0]) / (self.x_a.size - 1)
        self.z_a = self.field.z_a
        self.dz = (self.z_a[-1] - self.z_a[0]) / (self.z_a.size - 1)

        self.pressure = self.field.press
        self.salinity = self.field.xy_sa
        self.theta = self.field.xy_ct

        self.lp_cutoff = 50e3

        filter_keywords = dict(rp=0.1, rs=40, btype='lowpass', ftype='cheby2',
                               fs=1 / self.dx)

        # pole, zero filter allows for "Gust" treatment of edges
        self.sos_lp = iirfilter(17, 1 / self.lp_cutoff, output='sos',
                                **filter_keywords)
        self.b_lp, self.a_lp = iirfilter(7, 1 / self.lp_cutoff,
                                         **filter_keywords)

        self.sigma = self.field.xy_sig
        self.spice = self.field.xy_gamma
        self.c = self.field.xy_c

        self.lvls = lvl_profiles(self.z_a,
                                 self.sigma,
                                 self.spice,
                                 self.cf.sig_lvl)

        # last few contours should not have any nans
        cntr_nan = np.any(np.isnan(self.lvls[0, :, :]), axis=1)
        first_full = np.argmax(~cntr_nan)
        last_full = first_full + np.argmax(cntr_nan[np.argmax(~cntr_nan):])
        self.sig_lvl = self.cf.sig_lvl[:last_full]
        self.lvls = self.lvls[:, :last_full, :]

        # Climatology for region
        climatology = loadmat(climatology_file)
        self.z_clim = climatology['depth'].astype(np.float64).squeeze()
        clim_press = gsw.p_from_z(-self.z_clim, self.cf.lat)

        climi = 3  # climatolgy from April
        sal_psu = climatology['sal'].astype(np.float64)
        sal_clim = gsw.SA_from_SP(sal_psu, clim_press[:, None],
                                  self.cf.lon, self.cf.lat)
        self.sal_clim = sal_clim[:, climi]
        self.temp_clim = climatology['temp'].astype(np.float64)[:, climi]


    def stable_cntr_height(self, lvls):
        """Lowpass contours untill they become unstable, then truncate"""
        lp_lvls = np.zeros(lvls.shape)

        for tp0, tp1 in zip(self.cf.top_cntr[:-1], self.cf.top_cntr[1:]):
            # linear interpolation if datasat is too short for filter
            if (self.x_a[-1] - tp0[1] * self.dx) < 2 * self.lp_cutoff:
                orig = lvls[0, tp0[0]: tp1[0], tp0[1]:]
                proc = lp_lvls[0, tp0[0]: tp1[0], tp0[1]:]
                x = np.arange(tp0[1], self.x_a.size)
                for o, p in zip(orig, proc):
                    lr = linregress(x, o)
                    p = lr.intercept + lr.slope * x
            else:

                lp_lvls[0, tp0[0]: tp1[0], tp0[1]:] = \
                    filtfilt(self.b_lp, self.a_lp,
                             lvls[0, tp0[0]: tp1[0], tp0[1]:],
                             method="gust")

        # filter the rest of the contours
        lp_lvls[0, self.cf.top_cntr[-1][0]:, :] = \
            filtfilt(self.b_lp, self.a_lp,
                     lvls[0, self.cf.top_cntr[-1][0]:, :],
                     method="gust")

        all_diff = np.diff(lp_lvls[0, ::-1, :], axis=0)
        # filter to make lvls monotonic
        all_diff[all_diff > 0] = 0
        trapz = np.cumsum(all_diff, axis=0)[::-1, :]
        lp_lvls[0, :-1, :] = lp_lvls[0, -1, :] + trapz

        lp_lvls[lp_lvls < 1e-10] = 0.

        lp_lvls[1, :, :] = lvls[1, :, :]

        return lp_lvls


    def stable_spice(self, lvls, break_spice=False):
        """Compute estimate of stable spice"""
        stable_spice_lvls = lvls.copy()
        if self.spice_def == 2:
            # moving average used for spice definition
            #len_avg = 100  # needs to be even
            len_avg = 50  # needs to be even

            sig_mean_sa = []
            for i in range(stable_spice_lvls.shape[-1] - len_avg):
                sig_mean_sa.append(np.nanmean(stable_spice_lvls[1, :, i:i + len_avg], axis=1))

            o_s = (stable_spice_lvls.shape[1], len_avg // 2)
            sig_mean_sa = np.concatenate([np.broadcast_to(sig_mean_sa[0][:, None], o_s),
                                          np.array(sig_mean_sa).T,
                                          np.broadcast_to(sig_mean_sa[-1][:, None], o_s)], axis=1)
            stable_spice_lvls[1] = sig_mean_sa

        else:
            cntrs = np.moveaxis(stable_spice_lvls, 1, 0)
            if break_spice:
                for i, cntr in enumerate(cntrs):
                    if i in self.cf.break_inds:
                        brks = self.cf.breakpoints[self.cf.break_inds.index(i)]
                    else:
                        brks = None
                    self._lp_spice(cntr, brks)
            else:
                [self._lp_spice(cntr, None) for cntr in cntrs]

        return stable_spice_lvls


    def _lp_spice(self, contour, breaks):
        """Lowpass or take mean of spice value along contours"""

        nan_i = np.isnan(contour[1])
        if np.all(nan_i):
            # all points in contour are nan
            return

        if breaks is not None:
            breaks = set(breaks)

        seg_start = 0
        while seg_start < contour.shape[-1] - 1:

            seg_start += np.argmax(~nan_i[seg_start:])

            if np.any(nan_i[seg_start:]):
                seg_size = np.argmax(nan_i[seg_start:])
            else:
                seg_size = contour.shape[-1] - 1 - seg_start

            if seg_size == 0:
                # check for all nans at the end of contour
                return

            is_break = False
            if breaks is not None:
                inds = set(range(seg_start, seg_start + seg_size))
                bps = list(inds & breaks)
                bps.sort()
                for i in bps:
                    is_break = True
                    seg_size = i - seg_start + 1
                    self._lp_filter(contour, seg_start, seg_size)
                    seg_start = seg_start + seg_size
            if is_break:
                seg_size = max(inds) - seg_start + 1
            #else:
            self._lp_filter(contour, seg_start, seg_size)

            # move to next segment
            seg_start = seg_start + seg_size


    def _lp_filter(self, contour, seg_start, seg_size):
        """Perform a lp filter or linear fit depending on segment length"""
        seg_spice = contour[1, seg_start: seg_start + seg_size]

        if seg_spice.size == 1:
            # move to next segment
            return

        if seg_size * self.dx > 2 * self.lp_cutoff:
            filt_spice = filtfilt(self.b_lp, self.a_lp,
                                    seg_spice, method="gust")
            contour[1, seg_start: seg_start + seg_size] = filt_spice

        else:
            # linear fit to small sections
            x = np.arange(seg_spice.size)
            res = linregress(x, y=seg_spice)
            line = res.intercept + res.slope * x
            contour[1, seg_start: seg_start + seg_size] = line


    def compute_c_field(self, lvls, append_clim=True):
        """Create gridded sound speed field with climatolgy appended"""
        sigma, spice = grid_field(self.z_a, lvls, self.sig_lvl)
        sa, ct = self.field.sa_ct_from_sig_gamma(sigma, spice)

        if append_clim:
            z_clim_a, _, _, c = append_climatolgy(self.z_a, ct, sa,
                                                  self.z_clim, self.temp_clim,
                                                  self.sal_clim)
        else:
            c = gsw.sound_speed(sa, ct, self.pressure[:, None])
            z_clim_a = self.z_a
        return z_clim_a, c
