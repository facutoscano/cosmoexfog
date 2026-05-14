#%% Imports
import numpy as np

#%% Functions
def slice_cls(ell_cut, data_cl, data_std, ell_bins_edges, low_cut=False):
    """
    Cut power spectra based on lmin or lmax.
    If low_cut=True, cuts the low multipoles (l < ell_cut).
    If low_cut=False, cuts the high multipoles (l > ell_cut).

    Returns:
    - cut_cl: Cut power spectrum.
    - cut_std: Cut power spectrum standard deviation.
    - ell_bins_edges: Cut ell bin edges.
    """

    if ell_cut is None:
        return data_cl.copy(), data_std.copy(), ell_bins_edges.copy()
        
    bin_centers = (ell_bins_edges[:-1] + ell_bins_edges[1:]) / 2
    mask = (bin_centers >= ell_cut) if low_cut else (bin_centers <= ell_cut)
    
    cut_cl = data_cl[mask]
    cut_std = data_std[mask]
    idx_cut = np.where(mask)[0]
    
    if len(cut_cl) == 0: 
        return np.array([]), np.array([]), np.array([])
        
    return cut_cl, cut_std, ell_bins_edges[idx_cut[0] : idx_cut[-1]+2]

def bin_theory_spectrum(theo_dls, ell_bins_edges):
    """
    Bin the theoretical power spectrum D_l into specified ell bins.

    Returns:
    - theo_dls_bin: Binned theoretical power spectrum.
    """

    nbins = len(ell_bins_edges) - 1
    theo_dls_bin = np.zeros(nbins)
    for i in range(nbins):
        l_min_bin = int(ell_bins_edges[i])
        l_max_bin = int(ell_bins_edges[i+1])
        theo_dls_bin[i] = np.mean(theo_dls[l_min_bin:l_max_bin])
    return theo_dls_bin