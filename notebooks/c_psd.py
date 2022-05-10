import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import general_cosine

plt.ion()
fields = np.load('data/processed/decomposed_fields.npz')

x_a = fields['x_a']
z_a = fields['z_a']
c_bg = fields['c_bg']
c_tilt = fields['c_tilt']
c_spice = fields['c_spice']
c_total = fields['c_total']

lvls = np.load('data/processed/inputed_decomp.npz')
tot_lvls = lvls['filled_lvls']
stable_spice = lvls['stable_spice']
stable_lvls = lvls['stable_lvls']

#nfft = c_bg.shape[1] // 5
nfft = 100
nuttall_3b = [0.4243801, 0.4973406, 0.0782793]
window = general_cosine(nfft, nuttall_3b)
noverlap = int(nfft * 0.705)

z_ind = z_a < 150

f, tilt_psd = welch((c_tilt - c_bg)[z_ind, :], window=window,
                    noverlap=noverlap, axis=-1)
f, spice_psd = welch((c_spice - c_bg)[z_ind, :], window=window,
                     noverlap=noverlap, axis=-1)
f, total_psd = welch((c_total - c_bg)[z_ind, :], window=window,
                     noverlap=noverlap, axis=-1)

eps = np.spacing(1)

fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(6.5, 4))
ax[0].pcolormesh(np.log10(f + eps), z_a[z_ind], 10 * np.log10(f ** 2 * tilt_psd + eps), vmin=-40, vmax=-10)
ax[1].pcolormesh(np.log10(f + eps), z_a[z_ind], 10 * np.log10(f ** 2 * spice_psd + eps), vmin=-40, vmax=-10)
ax[2].pcolormesh(np.log10(f + eps), z_a[z_ind], 10 * np.log10(f ** 2 * total_psd + eps), vmin=-40, vmax=-10)
ax[0].set_ylim(150, 0)
ax[0].set_xlim(-2, -.3)


plot_i = 55
comp_i = 0
z_diff = tot_lvls[comp_i, plot_i, :] - stable_lvls[comp_i, plot_i, :]
z_i = ~(stable_lvls[comp_i, plot_i, :] == 0)

f, tilt_psd = welch(z_diff[z_i], window=window, noverlap=noverlap, axis=-1)
"""
fig, ax = plt.subplots(2, 1)
ax[0].plot(x_a / 1e3, tot_lvls[comp_i, plot_i, :])
ax[0].plot(x_a / 1e3, stable_lvls[comp_i, plot_i, :])
ax[0].plot(x_a[z_i] / 1e3, z_diff[z_i])

ax[1].plot(f, 10 * np.log10(tilt_psd))
"""
