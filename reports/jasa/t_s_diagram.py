import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src import Field

plt.ion()

field = Field(is_spiceness=False)

position = np.broadcast_to(field.x_a, field.xy_ct.shape)

norm = mpl.colors.Normalize(vmin=0, vmax=field.x_a[-1])
fig, ax = plt.subplots()
ax.scatter(field.xy_ct, field.xy_sa, c=position, cmap=plt.cm.cividis, norm=norm)

ax.set_xlabel(r'Tempurature, $\theta$ ($^\circ$C)')
ax.set_ylabel(r'Salinity, S$_{A}$ (g kg$^{-1}$)')

pos = ax.get_position()
pos.x0 += 0.05
pos.x1 += 0.05
pos.y0 += 0.05
pos.y1 += 0.05
ax.set_position(pos)


