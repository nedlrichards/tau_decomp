import numpy as np
from math import pi
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from copy import copy
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import find_peaks

from src import Config
import pickle

plt.ion()
cmap = copy(plt.cm.magma_r)
cmap.set_under('w')
bbox = dict(boxstyle='round', fc='w')

with open("data/processed/bg_loop_length.pic", "rb") as f:
    loop_length = pickle.load(f)

m_ll = []
m2_ll = []
for ll in loop_length:
    ll_sort = np.sort(ll)
    m_ll.append(ll_sort[-1])
    m2_ll.append(ll_sort[-2])

m_ll = np.array(m_ll)
m2_ll = np.array(m2_ll)

#fig, ax = plt.subplots()
#ax.plot(m_ll / 1e3, np.ones_like(m_ll), marker='|', linestyle="None")

#fig, ax = plt.subplots()
#ax.plot(m2_ll / 1e3, np.ones_like(m2_ll), marker='|', linestyle="None")

fig, ax = plt.subplots()
ax.plot(m_ll / 1e3)
ax.set_ylim(40, 400)

