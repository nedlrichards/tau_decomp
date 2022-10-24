import numpy as np
from math import pi
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from src import EngProc, Config, list_tl_files, MLEnergy
from src.eng_processing import field_stats

import matplotlib as mpl

plt.style.use('elr')
plt.ion()

#range_bounds = (7.5e3, 47.5e3)
range_bounds = (15e3, 47.5e3)
loaddir = 'data/processed/'
cf = Config()

tl_eng_400 = np.load(join(loaddir, 'int_eng_shallow_400.npz'))
tl_eng_1000 = np.load(join(loaddir, 'int_eng_shallow_1000.npz'))

r_a = tl_eng_400['r_a']
r_i = (r_a >= range_bounds[0]) & (r_a <= range_bounds[1])

tl_ref_1000 = (tl_eng_1000['ml_tl'][0])[None, :, r_i]
tl_db_1000 = (tl_eng_1000['ml_tl'][1:])[:, :, r_i]

tl_ref_400 = (tl_eng_400['ml_tl'][0])[None, :, r_i]
tl_db_400 = (tl_eng_400['ml_tl'][1:])[:, :, r_i]

dr = (r_a[-1] - r_a[0]) / (r_a.size - 1)
avg_len = 5e3
num_points = int(avg_len / dr)

run_avg = np.cumsum(tl_db_400, axis=-1)
#run_avg = np.cumsum(tl_db_400 - tl_ref_400, axis=-1)
run_avg[:, :, num_points:] = run_avg[:, :, num_points:] - run_avg[:, :, :-num_points]
run_avg = run_avg[:, :, num_points - 1:] / num_points
max_run_avg_400 = np.max(run_avg, axis=-1)

run_avg = np.cumsum(tl_db_1000, axis=-1)
#run_avg = np.cumsum(tl_db_1000 - tl_ref_1000, axis=-1)
run_avg[:, :, num_points:] = run_avg[:, :, num_points:] - run_avg[:, :, :-num_points]
run_avg = run_avg[:, :, num_points - 1:] / num_points
max_run_avg_1000 = np.max(run_avg, axis=-1)


fig, ax = plt.subplots()
ax.plot(max_run_avg_400[1])
ax.plot(max_run_avg_400[2])
ax.plot(max_run_avg_400[3])
ax.plot(max_run_avg_400[0])

ax.set_ylim(-30, -15)

#fig, ax = plt.subplots()
#ax.plot(max_run_avg_1000[1])
#ax.plot(max_run_avg_1000[2])
#ax.plot(max_run_avg_1000[3])
#ax.plot(max_run_avg_1000[0])
#
#ax.set_ylim(-30, -15)
