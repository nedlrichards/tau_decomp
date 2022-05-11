import numpy as np
import gsw
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Config
from src import Field

plt.ion()
cf = Config()
field = Field()

gamma = field.xy_gamma
sig = field.xy_sig
press = field.press[:, None]

c = field.c_from_sig_gamma(sig, gamma, press)
print(np.max(np.abs(c - field.xy_c)))
