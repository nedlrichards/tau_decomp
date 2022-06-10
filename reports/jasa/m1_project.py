import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

from src import MLEnergy, MLEnergyPE, list_tl_files, Config
from src import RDModes, section_cfield

plt.ion()
plt.style.use('elr')

fc = 400
tl_files = list_tl_files(fc=fc)
cf = Config(fc=fc)

percent_max = 99.99

def eng_diff(tl_file, field_type):
    tl_data = np.load(tl_file)
    p_field = tl_data['p_' + field_type]
    z_a = tl_data['zplot']
    r_a = tl_data['rplot']

    eng_mode = MLEnergy(tl_file, m1_percent=percent_max)
    ind = eng_mode.set_1[field_type]
    psi_1 = eng_mode.field_modes[field_type].psi_bg[ind]

    psi_ier = interp1d(tl_data['z_a'], np.squeeze(psi_1))
    psi_proj = psi_ier(z_a)
    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)
    z0_ind = np.argmax(np.abs(np.diff(np.sign(psi_proj))) > 1.5)

    total_eng = np.sum(np.abs(p_field[:, :z0_ind]) ** 2, axis=-1) * dz / 1e3
    proj_amp = np.sum(p_field[:, :z0_ind] * psi_proj[:z0_ind], axis=-1) * dz / 1e3

    norm_proj = 10 * np.log10(np.abs(proj_amp) ** 2 * (r_a - tl_data['xs']))
    norm_total = 10 * np.log10(total_eng * (r_a - tl_data['xs']))
    return norm_proj, norm_total

e_prj = []
e_tot = []
diff = []

for field_type in cf.field_types:
    e_p = []
    e_t = []
    d = []

    for tl_file in tl_files:
        proj, total = eng_diff(tl_file, field_type)
        e_p.append(proj)
        e_t.append(total)
        d.append(proj - total)

    e_prj.append(np.array(e_p))
    e_tot.append(np.array(e_t))
    diff.append(np.array(d))

e_prj = np.array(e_prj)
e_tot = np.array(e_tot)
diff = np.array(diff)

r_plot = []
for tl_file in tl_files:
    tl_data = np.load(tl_file)
    r_a = tl_data['rplot'] - tl_data['xs']
    r_plot.append(tl_data['xs'])
r_plot = np.array(r_plot)

r_bounds = (r_a > 10e3) & (r_a < 40e3)
m_diff = np.mean(diff[:, :, r_bounds], axis=-1)[:, :, None]
mode_en_var = np.var(diff[:, :, r_bounds] - m_diff, axis=-1)
mode_en_75 = np.percentile(diff[:, :, r_bounds] - m_diff, 75, axis=-1,
                           method='median_unbiased')

fig, ax = plt.subplots(figsize=(cf.jasa_1clm, 2.5))
ax.plot(r_plot / 1e3, mode_en_75[1:, :].T)
ax.plot(r_plot / 1e3, mode_en_75[0, :].T, color='0.4')
ax.set_ylim(-.1, 3)
ax.set_xlim(r_plot[0] / 1e3 -5, r_plot[-1] / 1e3 + 5)
labels = cf.field_types[1:]
labels.append(cf.field_types[0])
ax.legend(labels=labels, loc=2)
ax.set_xlabel('Source position, $x$ (km)')
ax.set_ylabel('Q$_3$ of mode 1 energy ratio (dB)')

ax.grid()

pos = ax.get_position()
pos.x0 += 0.05
pos.x1 += 0.06
pos.y0 += 0.08
pos.y1 += 0.05
ax.set_position(pos)

savedir = 'reports/jasa/figures'
fig.savefig(os.path.join(savedir, 'energy_variance.png'), dpi=300)
