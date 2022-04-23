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
        self.top_cntr = [[20, 759], [22, 707], [24, 615], [29, 563], [32, 515],
                         [34, 462], [36, 397], [41, 249], [44, 200], [48, 139],
                         [50, 62], [54, 0]]

        # pw spice model
        self.break_inds = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        self.breakpoints = [
            [918],
            [870, 918],
            [738, 917],
            [738, 917],
            [738, 916],
            [739, 915],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 914],
            [600, 687, 740, 915],
            [600, 687, 740, 915],
            [508, 600, 687, 741, 915],
            [508, 600, 687, 741, 915],
            [508, 600, 687, 741, 915],
            [508, 600, 687, 741, 915],
            [508, 600, 687, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [508, 600, 686, 742, 915],
            [137, 508, 600, 686, 741, 916],
            [137, 508, 600, 686, 741, 916],
            [137, 508, 600, 686, 741, 916],
            [137, 508, 600, 686, 741, 916],
            ]


