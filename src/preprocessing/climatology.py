import numpy as np
import gsw


def append_climatolgy(z_a, temp_grid, sal_grid, z_clim_a, temp_clim, sal_clim,
                      z_blend=320., lat=30.):
    """
    Linearly blend climatology to the bottom of sal and temp sections
    """
    dz = (z_a[-1] - z_a[0]) / (z_a.size - 1)

    z_up = np.arange(np.ceil(z_clim_a[-1] / dz)) * dz
    sal_up = np.interp(z_up, z_clim_a, sal_clim)
    temp_up = np.interp(z_up, z_clim_a, temp_clim)

    sal_merged = []
    temp_merged = []

    for sp, tp in zip(sal_grid.T, temp_grid.T):
        sal_merged.append(_profile_blend(sp, z_a, z_up, sal_up, z_blend))
        temp_merged.append(_profile_blend(tp, z_a, z_up, temp_up, z_blend))

    sal_merged = np.array(sal_merged).T
    temp_merged = np.array(temp_merged).T

    pressure = gsw.p_from_z(-z_up, lat)
    c_merged = gsw.sound_speed(sal_merged, temp_merged, pressure[:, None])

    return z_up, temp_merged, sal_merged, c_merged


def _profile_blend(section_profile, z_a, z_up, clim_profile, z_blend):
    """Blend a single profile to climatology"""

    merged_out = clim_profile.copy()

    # take care of nans if any
    not_nan = ~np.isnan(section_profile)
    sp = section_profile[not_nan]
    zp = z_a[not_nan]

    start_i = np.argmax(not_nan)

    copyi = np.argmax(~(zp <= z_blend)) + start_i

    merged_out[:copyi] = section_profile[:copyi]
    num_samples = zp.size - copyi + start_i

    # combine section and climatolgy with a linear blend
    lin_comb = np.arange(num_samples) / num_samples

    merge_section = merged_out[copyi:copyi + num_samples]
    merge_section *= lin_comb
    merge_section += (1 - lin_comb) * sp[copyi - start_i:]

    return merged_out
