from scipy.io import loadmat
import gsw
import numpy as np
import matplotlib.pyplot as plt

from src import Config, Section
from src import grid_field, SA_CT_from_sigma0_spiciness0

plt.ion()
cf = Config()

sec4 = Section()
z = sec4.lvls.copy()[0, :, :]
spice = sec4.lvls.copy()[1, :, :]

plot_i = [37, 40, 48]

#fig, ax = plt.subplots()
#ax.plot(sec4.x_a / 1e3, spice[plot_i].T)
#ax.plot(sec4.x_a / 1e3, np.sort(spice[plot_i])[::-1])

fig, ax = plt.subplots()
ax.plot(sec4.x_a / 1e3, z[plot_i].T)
ax.set_ylim(150, 0)
