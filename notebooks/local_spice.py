import numpy as np
import gsw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Config
from src import LocalSpice

plt.ion()
cf = Config()
spice = LocalSpice()

gamma = 1
sig = spice.sig_ref + 1
press = spice.p_ref

g_est, sig_est = spice.c_from_sig_gamma(sig, gamma, press)

print((sig_est - sig) / gamma)
print((gamma - g_est) / gamma)
