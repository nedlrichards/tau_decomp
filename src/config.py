from numpy import array, arange
from matplotlib.pyplot import cm
from copy import copy

class Config:
    """Common enviornment configuration"""

    def __init__(self, source_depth="shallow", fc=400.,
                 c_bounds=[1503., 1545.]):
        """define config"""
        self.fc = fc
        self.bottom_HS = [1600., 1000., 0.1]
        self.c_bounds = c_bounds

        if source_depth=="shallow":
            self.z_src = 40.
        elif source_depth == "deep":
            self.z_src = 200.
        else:
            raise(ValueError("Source depth must be either shallow or deep"))
        self.source_depth = source_depth

        self.z_int = 120.

        # plot specifications
        cmap = copy(cm.magma_r)
        cmap.set_under('w')
        self.cmap = cmap
        self.bbox = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
        self.jasa_1clm = 3.4  # 1 column figure width, in
        self.jasa_2clm = 7.05  # 2 column figure width, in

        self.field_types = ['bg', 'total', 'tilt', 'spice']

        self.lat = 30.
        self.lon = -140.

        # isopycnal spacings for sigma re 100 m
        d_iso = 0.01  # fine isopycnals spacing
        sig_start = 25.27
        sig_end = 26.99
        self.sig_lvl = arange(sig_start, sig_end, d_iso)

        # stable contours
        self.top_cntr = [[14, 923], [19, 905], [20, 757], [21, 710], [23, 618],
                         [26, 612], [28, 601], [30, 564], [31, 526], [33, 514],
                         [36, 460], [38, 418], [40, 402], [42, 268], [44, 205],
                         [48, 129], [50, 55],  [53, 0]]
