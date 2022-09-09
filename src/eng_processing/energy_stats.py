import numpy as np
from scipy.stats import linregress

def field_stats(r_a, field_eng, range_bounds=(5e3, 47e3)):
    """common statistics taken over field realization"""
    r_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])

    f_mean = np.mean(field_eng[:, r_i], axis=0)
    f_rms = np.sqrt(np.var(field_eng[:, r_i], axis=0))
    f_15 = np.percentile(field_eng[:, r_i], 15, axis=0,
                            method='median_unbiased')
    f_85 = np.percentile(field_eng[:, r_i], 85, axis=0,
                            method='median_unbiased')

    r_a = r_a[r_i]
    f_mean_rgs = linregress(r_a, y=f_mean)
    f_rms_rgs = linregress(r_a, y=f_rms)
    f_15_rgs = linregress(r_a, y=f_15)
    f_85_rgs = linregress(r_a, y=f_85)

    stats = {"r_a":r_a[r_i], "r_i":r_i, 'mean':f_mean, 'rms':f_rms,
                '15th':f_15, '85th':f_85,
                'mean_rgs':f_mean_rgs, 'rms_rgs':f_rms_rgs,
                '10th_rgs':f_15_rgs, '90th_rgs':f_85_rgs}
    return stats


def rgs(r_a, stat_type, stat_dict, range_bounds=(5e3, 47e3), scale_r=False):
    """compute linear regression line from object"""
    r_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])
    r_a = r_a[r_i]
    lin_rgs = stat_dict[stat_type + '_rgs']
    stat_fit = lin_rgs.intercept + r_a * lin_rgs.slope
    if scale_r:
        r_a /= 1e3
    return r_a, stat_fit
