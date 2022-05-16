from numpy import array, linspace, concatenate
from matplotlib.pyplot import cm
from copy import copy

class Config:
    """Common enviornment configuration"""

    def __init__(self, fc=400.):
        """define config"""
        self.fc = fc
        self.bottom_HS = [1600., 1000., 0.1]
        self.c_bounds = [1503, 1545]
        self.z_src = 40.
        self.z_int = 120.

        # plot specifications
        cmap = copy(cm.magma_r)
        cmap.set_under('w')
        self.cmap = cmap
        self.bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
        self.jasa_1clm = 3.4  # 1 column figure width, in
        self.jasa_2clm = 7.05  # 2 column figure width, in

        self.field_types = ['bg', 'tilt', 'spice', 'total']

        self.lat = 30.
        self.lon = -140.

        num_sig=[60, 40]
        sig_range = (24.80, 25.43, 26.60)
        high_samp = linspace(sig_range[0], sig_range[1], num_sig[0])
        low_samp = linspace(sig_range[1], sig_range[2], num_sig[1])[1:]
        self.sig_lvl = concatenate([high_samp,low_samp])

        # stable contours
        self.top_cntr = [[14, 923], [19, 905], [20, 757], [21, 710], [23, 618],
                         [21, 710], [23, 618], [26, 612], [28, 601], [30, 564],
                         [31, 526], [33, 514], [36, 460], [38, 418], [40, 402],
                         [42, 268], [44, 205], [48, 129], [50, 55],  [53, 0]]

