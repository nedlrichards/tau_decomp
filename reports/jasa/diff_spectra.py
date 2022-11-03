"""Compare results of gridded and reconstructed total field"""

import numpy as np
from scipy.io import loadmat
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.transforms
from copy import copy
import os

from src import sonic_layer_depth, list_tl_files, Config, section_cfield
from scipy.stats import linregress
from scipy.signal import welch, periodogram
from scipy.signal.windows import general_cosine


plt.ion()
plt.style.use('elr')
cf = Config()

fields = np.load('data/processed/inputed_decomp.npz')
x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_tilt = fields['c_tilt']
c_spice = fields['c_spice']
c_total = fields['c_total']
c_mean = np.mean(c_total, axis=1)

z_ml = np.array([20, 40, 60, 80])
z_tl = np.array([100, 120, 140, 160])
z_tmc = np.array([180, 200, 220, 240])

delta_bg = c_bg - c_mean[:, None]
delta_tilt = c_tilt - c_bg
delta_spice = c_spice - c_bg
delta_total = c_total - c_bg
#delta_total = c_total - c_mean[:, None]

def plot_spectra(z_avg, field, axis, *args, **kwargs):
    k_a = np.arange(x_a.size // 2 + 1) / x_a.size

    fft_i = np.argmin(np.abs(z_a[:, None] - z_avg), axis=0)
    spec_values = field[fft_i, :]
    #spec_fft = np.abs(np.fft.rfft(spec_values, axis=1)) ** 2
    #NFFT = 256
    NFFT = 128
    window = general_cosine(NFFT, [0.4243801, 0.4973406, 0.0782793])
    f, spec_fft = welch(spec_values, window=window, noverlap=int(NFFT * 0.705), axis=1)
    spec_mean = np.mean(spec_fft, axis=0)

    axis.loglog(f, spec_mean, *args, **kwargs)

    # fit line to spectra
    #k_cut = 1 / 50
    #k_fit = k_a > k_cut
    #rgs = linregress(np.log(k_a[k_fit]), np.log(spec_mean[k_fit]))
    #rgs = linregress(np.log(k_a[k_fit]), np.log(spec_mean[k_fit]))
    #axis.loglog(k_a[k_fit], np.exp(rgs.intercept + np.log(k_a[k_fit]) * rgs.slope), 'k', linewidth=3, zorder=10)
    #axis.loglog(k_a[k_fit], np.exp(rgs.intercept + np.log(k_a[k_fit]) * rgs.slope), '--', *args, zorder=11, **kwargs)
    #print(rgs.slope)



fig, axes = plt.subplots(1, 3, figsize=(cf.jasa_2clm, 2.75), sharey=True)

plot_spectra(z_ml, delta_total, axes[0], color='C0')
plot_spectra(z_ml, delta_tilt, axes[0], color='C1')
plot_spectra(z_ml, delta_spice, axes[0], color='C2')
plot_spectra(z_ml, delta_bg, axes[0], label='bg', color='C3')

axes[0].set_xlim(8e-3, 1)
axes[1].set_xlim(8e-3, 1)
axes[2].set_xlim(8e-3, 1)
axes[0].set_ylim(1e-3, 1e2)

plot_spectra(z_tl, delta_total, axes[1], color='C0')
plot_spectra(z_tl, delta_tilt, axes[1], color='C1')
plot_spectra(z_tl, delta_spice, axes[1], color='C2')
plot_spectra(z_tl, delta_bg, axes[1], label='bg', color='C3')

plot_spectra(z_tmc, delta_total, axes[2], color='C0')
plot_spectra(z_tmc, delta_tilt, axes[2], color='C1')
plot_spectra(z_tmc, delta_spice, axes[2], color='C2')
plot_spectra(z_tmc, delta_bg, axes[2], label='bg', color='C3')

#axes[2].legend(['total', 'tilt', 'spice', 'bg'], loc='upper right', bbox_to_anchor=(1.13, 1.08),
               #fontsize=9, labelspacing=0.2, handlelength=0.5, framealpha=1.0)
axes[2].legend(['total', 'tilt', 'spice', 'bg'])

axes[0].set_xticks([1e-2, 1e-1, 1])
axes[1].set_xticks([1e-1, 1])
axes[2].set_xticks([1e-1])

axes[0].text(5e-2, 5e1, '(a)', bbox=cf.bbox)
axes[1].text(5e-2, 5e1, '(b)', bbox=cf.bbox)
axes[2].text(8e-3, 5e1, '(c)', bbox=cf.bbox)

[ax.grid() for ax in axes]

fig.supxlabel('Wavenumber (cpkm)')
fig.supylabel('S$_{\delta c}$ (m/s/cpkm)')


#labs = axes[0].get_xticklabels()
#labs[-1].set_text('')
#axes[0].set_xticklabels(labs)
#axes[1].set_xticklabels(labs)
#axes[2].set_xticklabels(labs)

pos = axes[0].get_position()
pos.x0 += 0.00
pos.x1 += 0.04
pos.y0 += 0.08
pos.y1 += 0.08
axes[0].set_position(pos)

pos = axes[1].get_position()
pos.x0 += 0.01
pos.x1 += 0.05
pos.y0 += 0.08
pos.y1 += 0.08
axes[1].set_position(pos)

pos = axes[2].get_position()
pos.x0 += 0.02
pos.x1 += 0.06
pos.y0 += 0.08
pos.y1 += 0.08
axes[2].set_position(pos)

savedir = 'reports/jasa/figures'
fig.savefig(os.path.join(savedir, 'diff_spectra.png'), dpi=300)
