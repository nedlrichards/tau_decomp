import numpy as np
from src import RDModes, Config, sonic_layer_depth

from os.path import join

fc = 400
cf = Config(fc)


def resort(field_type, tl_data, sort_data):
    rd_modes = RDModes(tl_data['c_' + field_type], tl_data['x_a'], tl_data['z_a'],
                        cf.fc, cf.z_src, c_bounds=cf.c_bounds)
    # mixed layer mode order
    bg_sld, sld_i = sonic_layer_depth(rd_modes.z_a, rd_modes.bg_prof[:, None], z_max=300.)
    ml_eng = np.sum(np.abs(rd_modes.psi_bg[:, rd_modes.z_a <= bg_sld]) ** 2, axis=-1)
    mode_order = np.argsort(ml_eng)[::-1]
    resort_i = np.argsort(mode_order)

    sort_data[field_type + "_mode_amps"] = tl_data[field_type + "_mode_amps"][:, resort_i]

def resort_npz(xs):
    run_number = int(xs)
    tl_data = np.load(join(f'data/processed/field_{int(fc)}',
                        f'tl_section_{run_number}.npz'))
    sort_data = dict(tl_data)

    field_types = ['bg', 'tilt', 'spice', 'total']
    [resort(ft, tl_data, sort_data) for ft in field_types]

    np.savez(join(f"data/processed/field_{int(fc)}",
                f"xmission_{run_number:03d}.npz"), **sort_data)

[resort_npz(xs * 10) for xs in range(91)]

