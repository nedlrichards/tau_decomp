"""Overview plots of transcet"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from os.path import join

from src import Section, sonic_layer_depth, grid_field, Config

plt.ion()
bbox = dict(boxstyle='round', fc='w')
savedir = 'reports/jasa/figures'

cf = Config()

sec4 = Section()

dtau_dsig = np.diff(sec4.spice, axis=0) / np.diff(sec4.sigma0, axis=0)

plot_i = 249
plot_i = 252

fig, ax = plt.subplots(1, 2, sharey=True)

ax[0].plot(dtau_dsig[:, plot_i], sec4.z_a[:-1])
ax[1].plot(sec4.sigma0[:, plot_i], sec4.z_a)

ax[0].set_ylim(150, 0)

