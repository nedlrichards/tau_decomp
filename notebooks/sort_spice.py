from scipy.io import loadmat
import gsw
import numpy as np
import matplotlib.pyplot as plt

from src import Config, lvl_profiles, Section
from src import grid_field, SA_CT_from_sigma0_spiciness0

plt.ion()
cf = Config()

sec4 = Section()
spice = sec4.lvls.copy()[1, :, :]

fig, ax = plt.subplots()
plot_i = [40, 45, 50]
ax.plot(sec4.x_a / 1e3, spice[plot_i].T)
#ax.plot(sec4.x_a / 1e3, np.sort(spice[plot_i])[::-1])
