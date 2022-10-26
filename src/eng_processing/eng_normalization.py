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


    def diffraction_bg(self, int_layer='ml', normalize=True):
        """Energy in range independent background duct"""
        eng_bg = []
        for ml in self.mls:
            bg_eng = ml.background_diffraction('bg', int_layer=int_layer, normalize=normalize)
            eng_bg.append(10 * np.log10(bg_eng * ml.r_a))

        return np.array(eng_bg)


    def dynamic_energy(self, field_types=None, mode_num=None, int_layer='ml'):
        """Compute range dependent energy in surface duct"""
        if field_types is None:
            field_types = self.cf.field_types

        fields = {f:[] for f in field_types}
        for fld in field_types:
            for ml in self.mls:
                if mode_num is None:
                    eng_db = 10 * np.log10(ml.ml_energy(fld, int_layer=int_layer) * ml.r_a)
                else:
                    proj_amp, proj_scale = ml.proj_mode(fld, mode_num=mode_num)
                    proj_eng = np.abs(proj_amp) ** 2 / proj_scale
                    eng_db = 10 * np.log10(proj_eng)

                fields[fld].append(eng_db)


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
