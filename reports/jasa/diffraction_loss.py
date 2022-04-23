import numpy as np
from math import pi
import matplotlib.pyplot as plt

import pickle

from src import Config

plt.ion()

fc = 400
z_int = 150.
cf = Config(fc)

bg_ri_eng = np.load('data/processed/bg_ri_eng.npz')

diff_eng = bg_ri_eng['diff_eng']
diff_eng_0 = bg_ri_eng['diff_eng_0']
r_a = bg_ri_eng['r_a']

eng_dB = 10 * np.log10(diff_eng * r_a).T
eng_dB_0 = 10 * np.log10(diff_eng_0 * r_a).T

fig, ax = plt.subplots()
ax.plot(r_a / 1e3, eng_dB - eng_dB_0, '0.4', linewidth=0.5)
ax.grid()

fig, ax = plt.subplots()
ax.plot(np.max(eng_dB_0, axis=0) - np.min(eng_dB_0, axis=0))
ax.grid()

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

fig, ax = plt.subplots()
ax.plot(m_ll / 1e3)
ax.set_ylim(40, 400)
ax.grid()

