#%% Imports
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

#%% Functions
def smooth_map(map_in, fwhm_deg):
    """
    Smooth a map given a FWHM [degrees].
    
    Returns:
    -Smoothed map.
    """

    fwhm_rad = np.radians(fwhm_deg)
    return hp.smoothing(map_in, fwhm=fwhm_rad, pol=False)


def project_voids_to_map(voids_df, nside, cmb_mask, frac_rvoid=1.0):
    """
    Project a void catalog (DataFrame) onto a Healpix map.
    The void catalog must contain the following columns:
    - 'l': Galactic longitude
    - 'b': Galactic latitude
    - 'R': Void radius
    - 'redshift': Void redshift

    Returns:
    - Map of hits.
    - Valid pixel mask.
    - Selection indices.
    """

    h = cosmo.H0.value / 100.0
    m = np.zeros(hp.nside2npix(nside))
    
    for _, void in voids_df.iterrows():
        l, b = void['l'], void['b']
        r_comovil_mpc = (void['R'] / h) * frac_rvoid * u.Mpc
        d_m = cosmo.comoving_transverse_distance(void['redshift'])
        r_angle = np.arcsin(r_comovil_mpc / d_m).value
    
        query = hp.query_disc(nside, hp.ang2vec(l, b, lonlat=True), r_angle)
        valid_pix = query[cmb_mask[query] == 1]
        m[valid_pix] += 1
        
    select_indices = (m > 0) & (cmb_mask == 1)
    valid_mask = np.zeros(hp.nside2npix(nside))
    valid_mask[select_indices] = 1.0
    return m, valid_mask, select_indices


def remove_low_multipoles(map_in, nside, l_rm, lmax):
    """Remove low multipoles. Handles nside limits automatically."""
    nan_mask = np.isnan(map_in)
    map_safe = np.nan_to_num(map_in, nan=0.0) if nan_mask.any() else map_in
    
    # Healpy strict mathematical limit to prevent crashes
    safe_lmax = min(lmax, 3 * nside - 1)
    
    alm = hp.map2alm(map_safe, lmax=safe_lmax)
    for j in range(min(l_rm + 1, safe_lmax + 1)):
        for k in range(j + 1):
            alm[hp.Alm.getidx(safe_lmax, j, k)] = 0
            
    map_removed = hp.alm2map(alm, nside, lmax=safe_lmax)
    if nan_mask.any():
        map_removed[nan_mask] = np.nan
        
    return map_removed