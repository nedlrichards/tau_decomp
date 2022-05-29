import numpy as np
import os
import matplotlib.pyplot as plt

from src import Config
from modepy import PRModes

plt.ion()

fc = 400
cf = Config(fc=fc)

c_fields = np.load('data/processed/inputed_decomp.npz')
z_a = c_fields['z_a']
c_prof = c_fields['c_bg'][:, 0]

modes = PRModes(z_a, c_prof, fc)
kr, psi = modes(cf.c_bounds)

fig, ax = plt.subplots()
ax.plot(psi[:, -50], z_a)

ax.set_ylim(150, 0)
