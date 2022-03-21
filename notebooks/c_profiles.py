import numpy as np
import matplotlib.pyplot as plt

from src import sonic_layer_depth

bbox=dict(boxstyle="round", fc="w")
plt.ion()

#save_dir = {'x_a':sec4.x_a, 'z_a':z_a, 'c_bg':c_bg, 'c_spice':c_spice,
            #'c_tilt':c_tilt, 'c_total':c_total}
c_fields = np.load('data/processed/decomposed_fields.npz')

field_type = 'total'

z_max = 200

z_a = c_fields['z_a']
x_a = c_fields['x_a']

z_i = z_a <= z_max

c_plot = (c_fields['c_' + field_type] - c_fields['c_bg'])[z_i, :]
z_a_p = z_a[z_i]

#fig, ax = plt.subplots()
#cm = ax.pcolormesh(x_a / 1e3, z_a_p, c_plot, cmap=plt.cm.coolwarm)
#fig.colorbar(cm)
#ax.set_ylim(150, 0)
#ax.set_xlim(150, 300)

# mean and variance

def prof_rms(x_bounds):
    if x_bounds is None:
        x_i = np.ones(x_a.size, dtype=np.bool_)
    else:
        x_i = (x_a >= x_bounds[0]) & (x_a <= x_bounds[1])

    c_bg = c_fields['c_bg']
    mean = np.mean(c_fields['c_total'][:, x_i], axis=-1)

    z_sld = z_a_p[np.argmax(mean[z_i])]
    bg_rms = np.sqrt(np.var(c_bg[:, x_i], axis=-1))

    tilt_rms = np.sqrt(np.var((c_fields['c_tilt'] - c_bg)[:, x_i], axis=-1))
    spice_rms = np.sqrt(np.var((c_fields['c_spice'] - c_bg)[:, x_i], axis=-1))
    total_rms = np.sqrt(np.var((c_fields['c_total'] - c_bg)[:, x_i], axis=-1))

    return z_sld, mean, np.array([tilt_rms, spice_rms, total_rms])

x_bounds = None

fig, ax = plt.subplots(1, 5, sharey=True, figsize=(6.5, 3))

for i in range(-1, 3):

    if i < 0:
        x_bounds = None
    else:
        x_bounds = np.array((0, 330e3))
        x_bounds += i * 330e3


    z_sld, mean, rms = prof_rms(x_bounds)

    if i == -1:
        ax[0].plot(mean[z_i], z_a_p, 'k')

    #ax[1].plot(bg_rms[z_i], z_a_p, '0.5')
    ax[i + 2].plot([-10, 10], [z_sld, z_sld], color='0.4', linestyle=":")
    ax[i + 2].plot(rms[0, z_i], z_a_p)
    ax[i + 2].plot(rms[1, z_i], z_a_p)
    ax[i + 2].plot(rms[2, z_i], z_a_p, 'k')

    ax[i + 2].grid()
    ax[i + 2].set_xlim(-0.1, 2.5)


ax[0].grid()
ax[0].set_ylim(200, 0)
#ax[0].text(11, 225, '+1500')
ax[1].text(4.5, 232, 'RMS (m/s)')
ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('Mean (m/s)')
#ax[1].set_xlabel('RMS (m/s)')

ax[0].text(1497, 12, '(a)', bbox=bbox)
ax[1].text(1.8, 12, '(b)', bbox=bbox)
ax[2].text(1.8, 12, '(c)', bbox=bbox)
ax[3].text(1.8, 12, '(d)', bbox=bbox)
ax[4].text(1.8, 12, '(e)', bbox=bbox)

pos = ax[0].get_position()
pos.x0 -= 0.02
pos.x1 -= 0.04
pos.y1 += 0.10
pos.y0 += 0.04
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 -= 0.04
pos.x1 -= 0.00
pos.y1 += 0.10
pos.y0 += 0.04
ax[1].set_position(pos)

pos = ax[2].get_position()
pos.x0 -= 0.00
pos.x1 += 0.02
pos.y1 += 0.10
pos.y0 += 0.04
ax[2].set_position(pos)

pos = ax[3].get_position()
pos.x0 += 0.02
pos.x1 += 0.04
pos.y1 += 0.10
pos.y0 += 0.04
ax[3].set_position(pos)

pos = ax[4].get_position()
pos.x0 += 0.04
pos.x1 += 0.06
pos.y1 += 0.10
pos.y0 += 0.04
ax[4].set_position(pos)

fig.savefig('reports/figures/mean_rms_prof_section.png', dpi=300)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(3, 3))

x_bounds = None
z_sld, mean, rms = prof_rms(x_bounds)

ax[0].plot(mean[z_i], z_a_p, 'k')

ax[1].plot([-10, 10], [z_sld, z_sld], color='0.4', linestyle=":")
ax[1].plot(rms[0, z_i], z_a_p)
ax[1].plot(rms[1, z_i], z_a_p)
ax[1].plot(rms[2, z_i], z_a_p, 'k')

ax[0].grid()
ax[1].grid()
ax[1].set_xlim(-0.1, 2.5)

ax[0].set_ylim(200, 0)
ax[1].text(1.5, 70, 'SLD', bbox=bbox)
ax[0].set_ylabel('Depth (m)')
ax[0].set_xlabel('Mean (m/s)')
ax[1].set_xlabel('RMS (m/s)')

ax[0].text(1497, 12, '(a)', bbox=bbox)
ax[1].text(1.8, 12, '(b)', bbox=bbox)

pos = ax[0].get_position()
pos.x0 += 0.07
pos.x1 += 0.09
pos.y1 += 0.10
pos.y0 += 0.04
ax[0].set_position(pos)

pos = ax[1].get_position()
pos.x0 += 0.07
pos.x1 += 0.09
pos.y1 += 0.10
pos.y0 += 0.04
ax[1].set_position(pos)

fig.savefig('reports/figures/mean_rms_prof.png', dpi=300)

