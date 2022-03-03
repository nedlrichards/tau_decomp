import numpy as np

def lvl_profiles(z_axis, sigma_grid, tau_grid, sig_lvls):
    """Represent sigma and spice profiles as discrete points"""
    all_profiles = []

    num_profs = sigma_grid.shape[-1]
    for i in range(num_profs):
        z_lvl = np.interp(sig_lvls, sigma_grid[:, i], z_axis,
                          left=np.nan, right=np.nan)
        spice_lvl = np.interp(z_lvl, z_axis, tau_grid[:, i],
                              left=np.nan, right=np.nan)
        all_profiles.append(np.array([z_lvl, spice_lvl])[:, :, None])

    all_profiles = np.concatenate(all_profiles, axis=-1)
    return all_profiles


def grid_field(z_axis, lvls, sig_lvls):
    """Interpolate between contour points to reconstruct field"""
    grid_sigma = []
    grid_tau = []
    for i in range(lvls.shape[-1]):

        # need to remove nans before interpolation
        lvl_i = ~np.isnan(lvls[0, :, i]) & (lvls[0, :, i] > 0)

        # interpolate density to a grid
        fill_left = sig_lvls[lvl_i][0]
        fill_right = sig_lvls[lvl_i][-1]
        sig_intp = np.interp(z_axis, lvls[0, lvl_i, i], sig_lvls[lvl_i],
                             left=fill_left, right=fill_right)
        grid_sigma.append(sig_intp)

        # interpolate spice to a grid
        fill_left = lvls[1, lvl_i, i][0]
        fill_right = lvls[1, lvl_i, i][-1]
        spice_intp = np.interp(z_axis, lvls[0, lvl_i, i], lvls[1, lvl_i, i],
                               left=fill_left, right=fill_right)
        # trying other values to fill in the top layer above last isopycnal
        nan_i = np.argmax(~np.isnan(spice_intp))
        fill_i = np.isnan(lvls[1, :, i]) | (lvls[1, :, i] == 0)
        #fill_val = np.median(sig_lvls[:np.argmax(~np.isnan(lvls[0, :, i]))+1])
        fill_val = lvls[1, [np.argmax(~fill_i)], i]
        spice_intp[: nan_i] = fill_val
        grid_tau.append(spice_intp)

    return np.array(grid_sigma).T, np.array(grid_tau).T


