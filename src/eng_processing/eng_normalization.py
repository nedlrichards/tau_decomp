import numpy as np
from os.path import join

from src import MLEnergyPE, Config, list_tl_files

class EngNorm:
    """common processing for mixed layer energy processing"""
    def __init__(self, fc, source_depth):
        """parameters of investigation grid"""
        self.fc = fc
        self.source_depth = source_depth
        self.cf = Config(fc=fc, source_depth=source_depth)

        # load background field energy for a reference
        eng_bg = []
        for tl in list_tl_files(fc, source_depth=source_depth):
            ml_pe = MLEnergyPE(tl)
            eng_bg.append(10 * np.log10(ml_pe.ml_energy('bg') * ml_pe.r_a))

        self.bg_eng = np.array(eng_bg)
        self.r_a = ml_pe.r_a

        # load dynamic fields
        dy_fields = self.cf.field_types.copy()
        dy_fields.remove('bg')

        for tl in list_tl_files(fc, source_depth=source_depth):
            ml_pe = MLEnergyPE(tl)
            for fld in dy_fields:
                fields[fld].append(10 * np.log10(ml_pe.ml_energy(fld) * ml_pe.r_a))

        ml_eng = []

        for fld in dy_fields:
            ml_eng.append(np.array(fields[fld]))
        self.dynamic_eng = np.array(ml_eng)

    def blocking_feature(self, range_bounds, integration_length=5, block_co=3):
        """Compute integrated loss indices blocking features"""
        dr = (self.r_a[-1] - self.r_a[0]) / (self.r_a.size - 1)
        num_int = int(np.ceil(self.integration_length * 1e3 / dr))

        diff_eng = self.dynamic_eng - self.bg_eng

        move_sum = np.cumsum(diff_eng, dtype=float, axis=-1)
        # integration with a size num_int moving window
        move_sum[:, :, num_int:] = move_sum[:, :, num_int:] \
                                - move_sum[:, :, :-num_int]
        move_sum = move_sum[:, :, win_len - 1:] * dr

        # max integrated loss
        max_int = np.max(-move_sum, axis=-1)
        return max_int




fc = 400
#fc = 1e3

source_depth = "shallow"
#source_depth = "deep"

cf = Config(source_depth=source_depth, fc=fc)

pe_ml_engs = []

x_s = []
all_eng = []
for r in list_tl_files(fc, source_depth=source_depth):
    e = MLEnergyPE(r)
    x_s.append(e.xs)
    o_r = np.array([e.ml_energy(ft) for ft in cf.field_types])
    all_eng.append(o_r[:, None, :])

r_a = e.r_a
x_s = np.array(x_s)

all_eng = np.concatenate(all_eng, axis=1)

norm_eng = np.log10(all_eng[1:, :, :]) - np.log10(all_eng[0, :, :])
norm_eng *= 10

dr = (r_a[-1] - r_a[0]) / (r_a.size - 1)
diff_i = (r_a > 5e3) & (r_a < 45e3)

# strange transpose arrises from indexing
diff_eng = np.diff(norm_eng[:, :, diff_i], axis=-1) / dr

win_len = 50
move_sum = np.cumsum(diff_eng, dtype=float, axis=-1)
move_sum[:, :, win_len:] = move_sum[:, :, win_len:] - move_sum[:, :, :-win_len]
move_sum = move_sum[:, :, win_len - 1:] * dr

max_int = np.max(-move_sum, axis=-1)

fig, ax = plt.subplots(figsize=(cf.jasa_1clm,2.5))
ax.plot(x_s / 1e3, max_int.T)
ax.set_xlabel('Starting position (km)')
ax.set_ylabel('Maxium loss over 5 km (dB)')
ax.set_ylim(-0.5, 15)
ax.set_xlim(0, 900)
ax.grid()
#ax.legend(['tilt', 'spice', 'observed'])
ax.set_yticks([0, 3, 6, 9, 12])
pos = ax.get_position()
pos.x0 += 0.04
pos.x1 += 0.05
pos.y0 += 0.07
pos.y1 += 0.07
ax.set_position(pos)

fig.savefig('reports/jasa/figures/integrated_loss.png', dpi=300)
