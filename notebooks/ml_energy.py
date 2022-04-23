import numpy as np
from math import pi
import matplotlib.pyplot as plt

import os

from src import MLEnergy, list_tl_files

#plt.ion()

fc = 400
tl_files = list_tl_files(fc)

def eng_partition(tl_path):
    """Partition energy into mode sets"""

    ml_eng = MLEnergy(fc, tl_path)

    en_ri, en_ri_0 = ml_eng.background_diffraction()

    en_bg = ml_eng.field_ml_eng("bg")
    en_tilt = ml_eng.field_ml_eng("tilt")
    en_spice = ml_eng.field_ml_eng("spice")
    en_total = ml_eng.field_ml_eng("total")

    en_bg_1 = ml_eng.field_ml_eng("bg", indicies=ml_eng.set_1["bg"])
    en_tilt_1 = ml_eng.field_ml_eng("tilt", indicies=ml_eng.set_1["tilt"])
    en_spice_1 = ml_eng.field_ml_eng("spice", indicies=ml_eng.set_1["spice"])
    en_total_1 = ml_eng.field_ml_eng("total", indicies=ml_eng.set_1["total"])

    en_bg_2 = ml_eng.field_ml_eng("bg", indicies=ml_eng.set_2["bg"])
    en_tilt_2 = ml_eng.field_ml_eng("tilt", indicies=ml_eng.set_2["tilt"])
    en_spice_2 = ml_eng.field_ml_eng("spice", indicies=ml_eng.set_2["spice"])
    en_total_2 = ml_eng.field_ml_eng("total", indicies=ml_eng.set_2["total"])

    ref_dB = 10 * np.log10(en_ri)
    ref_dB_0 = 10 * np.log10(en_ri_0)
    ref = ref_dB_0

    r_a = ml_eng.r_a + ml_eng.xs
    #r_i = (ml_eng.r_a > 5e3) & (ml_eng.r_a < 45e3)
    r_i = np.ones(ml_eng.r_a.size, dtype=np.bool_)


    fig, ax = plt.subplots()
    #ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_bg) - ref)[r_i])
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_tilt) - ref)[r_i])
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_spice) - ref)[r_i])
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_total) - ref)[r_i])

    ax.set_prop_cycle(None)

    #ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_bg_1) - ref)[r_i], '--')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_tilt_1) - ref)[r_i], '--')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_spice_1) - ref)[r_i], '--')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_total_1) - ref)[r_i], '--')

    ax.set_prop_cycle(None)
    #ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_bg_2) - ref)[r_i], ':')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_tilt_2) - ref)[r_i], ':')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_spice_2) - ref)[r_i], ':')
    ax.plot(r_a[r_i] / 1e3, (10 * np.log10(en_total_2) - ref)[r_i], ':')

    ax.grid()
    ax.set_ylim(-20, 5)
    #ax.set_xlim(ml_eng.xs / 1e3, ml_eng.xs / 1e3 + 55)

    filename = 'red_amp_' + tl_path.split('.')[0].split('_')[-1] + '.png'
    fig.savefig('reports/figures/red_amp/' + filename, dpi=300)
    plt.close(fig)

[eng_partition(tl) for tl in tl_files]
