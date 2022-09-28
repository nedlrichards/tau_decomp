import numpy as np
from os.path import join
from scipy.stats import linregress
from src import MLEnergy, Config, list_tl_files

class EngProc:
    """common processing for mixed layer energy processing"""
    def __init__(self, cf, fields=None):
        """parameters of investigation grid"""
        self.cf = cf

        # load background field energy for a reference
        xs = []
        mls = []

        for tl in list_tl_files(cf.fc, source_depth=cf.source_depth):
            ml = MLEnergy(tl, self.cf.source_depth, fields=fields)
            mls.append(ml)
            xs.append(ml.xs)

        self.xs = np.array(xs)
        self.r_a = ml.r_a
        self.mls = mls


    def diffraction_bg(self):
        """Energy in range independent background duct"""
        eng_bg = []
        for ml in self.mls:
            bg_eng = ml.background_diffraction('bg')
            eng_bg.append(10 * np.log10(bg_eng * ml.r_a))

        return np.array(eng_bg)


    def dynamic_energy(self, field_types=None):
        """Compute range dependent energy in surface duct"""
        if field_types is None:
            field_types = self.cf.field_types

        fields = {f:[] for f in field_types}
        for fld in field_types:
            for ml in self.mls:
                fields[fld].append(10 * np.log10(ml.ml_energy(fld) * ml.r_a))
            fields[fld] = np.array(fields[fld])

        return fields


    def blocking_feature(self, dynamic_eng, bg_eng, range_bounds=(5e3, 50e3),
                         comp_len=5e3):
        """Compute integrated loss indices blocking features"""
        dr = (self.r_a[-1] - self.r_a[0]) / (self.r_a.size - 1)
        num_int = int(np.ceil(comp_len / dr))

        diff_eng = dynamic_eng - bg_eng
        r_i = (self.r_a > range_bounds[0]) & (self.r_a < range_bounds[1])
        diff_eng = diff_eng[:, :, r_i]

        # cumulative loss
        loss = diff_eng[:, :, num_int:] - diff_eng[:, :, :-num_int]
        max_int = np.max(-loss, axis=-1)

        return max_int


    #def field_stats(self, field_eng, range_bounds=(5e3, 50e3)):
        #"""common statistics taken over field realization"""
        #r_i = (self.r_a >= range_bounds[0]) & (self.r_a <= range_bounds[1])
#
        #f_mean = np.mean(field_eng[:, r_i], axis=0)
        #f_rms = np.sqrt(np.var(field_eng[:, r_i], axis=0))
        #f_15 = np.percentile(field_eng[:, r_i], 15, axis=0,
                             #method='median_unbiased')
        #f_85 = np.percentile(field_eng[:, r_i], 85, axis=0,
                             #method='median_unbiased')
#
        #r_a = self.r_a[r_i]
        #f_mean_rgs = linregress(r_a, y=f_mean)
        #f_rms_rgs = linregress(r_a, y=f_rms)
        #f_15_rgs = linregress(r_a, y=f_15)
        #f_85_rgs = linregress(r_a, y=f_85)
#
        #stats = {"r_a":self.r_a[r_i], "r_i":r_i, 'mean':f_mean, 'rms':f_rms,
                 #'15th':f_15, '85th':f_85,
                 #'mean_rgs':f_mean_rgs, 'rms_rgs':f_rms_rgs,
                 #'10th_rgs':f_15_rgs, '90th_rgs':f_85_rgs}
        #return stats

##
    #def rgs(self, stat_type, stat_dict, range_bounds=(5e3, 50e3), scale_r=False):
        #"""compute linear regression line from object"""
        #r_i = (self.r_a >= range_bounds[0]) & (self.r_a <= range_bounds[1])
        #r_a = self.r_a[r_i]
        #lin_rgs = stat_dict[stat_type + '_rgs']
        #stat_fit = lin_rgs.intercept + r_a * lin_rgs.slope
        #if scale_r:
            #r_a /= 1e3
        #return r_a, stat_fit
