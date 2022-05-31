import numpy as np
from os.path import join
from scipy.stats import linregress
from src import MLEnergy, MLEnergyPE, Config, list_tl_files

class EngProc:
    """common processing for mixed layer energy processing"""
    def __init__(self, cf):
        """parameters of investigation grid"""
        self.cf = cf

        # load background field energy for a reference
        eng_bg = []
        xs = []

        for tl in list_tl_files(cf.fc, source_depth=cf.source_depth):
            #ml_pe = MLEnergyPE(tl)
            #eng_bg.append(10 * np.log10(ml_pe.ml_energy('bg') * ml_pe.r_a))
            #xs.append(ml_pe.xs)
            ml = MLEnergy(tl, self.cf.source_depth, bg_only=True)
            bg_eng, _ = ml.background_diffraction()
            eng_bg.append(10 * np.log10(bg_eng * ml.r_a))
            xs.append(ml.xs)

        self.xs = np.array(xs)
        self.bg_eng = np.array(eng_bg)
        self.r_a = ml.r_a

        # load dynamic fields
        dy_fields = self.cf.field_types.copy()
        dy_fields.remove('bg')

        fields = {f:[] for f in dy_fields}

        for tl in list_tl_files(cf.fc, source_depth=cf.source_depth):
            ml_pe = MLEnergyPE(tl)
            for fld in dy_fields:
                fields[fld].append(10 * np.log10(ml_pe.ml_energy(fld) * ml_pe.r_a))

        ml_eng = []

        for fld in dy_fields:
            ml_eng.append(np.array(fields[fld]))
        self.dynamic_eng = np.array(ml_eng)
        self.dy_fields = dy_fields


    def blocking_feature(self, range_bounds=(5e3, 50e3), comp_len=5e3):
        """Compute integrated loss indices blocking features"""
        dr = (self.r_a[-1] - self.r_a[0]) / (self.r_a.size - 1)
        num_int = int(np.ceil(comp_len / dr))

        diff_eng = self.dynamic_eng - self.bg_eng
        r_i = (self.r_a > range_bounds[0]) & (self.r_a < range_bounds[1])
        diff_eng = diff_eng[:, :, r_i]

        # cumulative loss
        loss = diff_eng[:, :, num_int:] - diff_eng[:, :, :-num_int]
        max_int = np.max(-loss, axis=-1)

        return max_int


    def field_stats(self, field_eng, range_bounds=(5e3, 50e3)):
        """common statistics taken over field realization"""
        r_i = (self.r_a > range_bounds[0]) & (self.r_a < range_bounds[1])

        f_mean = np.mean(field_eng[:, r_i], axis=0)
        f_rms = np.sqrt(np.var(field_eng[:, r_i], axis=0))
        f_10 = np.percentile(field_eng[:, r_i], 10, axis=0,
                             method='median_unbiased')
        f_90 = np.percentile(field_eng[:, r_i], 90, axis=0,
                             method='median_unbiased')

        r_a = self.r_a[r_i]
        f_mean_rgs = linregress(r_a, y=f_mean)
        f_rms_rgs = linregress(r_a, y=f_mean + f_rms)
        f_10_rgs = linregress(r_a, y=f_10)
        f_90_rgs = linregress(r_a, y=f_90)

        stats = {"r_a":self.r_a[r_i], 'mean':f_mean, 'rms':f_rms,
                 '10th':f_10, '90th':f_90,
                 'mean_rgs':f_mean_rgs, 'rms_rgs':f_rms_rgs,
                 '10th_rgs':f_10_rgs, '90th_rgs':f_90_rgs}
        return stats


    def rgs(self, stat_type, stat_dict, range_bounds=(5e3, 50e3), scale_r=False):
        """compute linear regression line from object"""
        r_i = (self.r_a > range_bounds[0]) & (self.r_a < range_bounds[1])
        r_a = self.r_a[r_i]
        lin_rgs = stat_dict[stat_type + '_rgs']
        stat_fit = lin_rgs.intercept + r_a * lin_rgs.slope
        if scale_r:
            r_a /= 1e3
        return r_a, stat_fit

