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

def build_ell_edges_and_slice_data(cl_matrix, cl_std, ell_min_file, ell_min,
                                    delta_ell):
    """
    Build ell_edges from the FITS file structure and trim the low-ell bins
    that the user wants to exclude from the analysis (handled by another code).

    Parameters
    ----------
    cl_matrix : ndarray
        Shape (n_realizations, n_bins_file). Untouched, only its shape is used.
    cl_std : ndarray
        Shape (n_bins_file,). Will be sliced consistently.
    ell_min_file : int
        First ell edge stored in the FITS file (typically 2).
    ell_min : int
        First ell edge to keep in the analysis. Bins whose center is below
        ell_min are discarded.
    delta_ell : int
        Bin width.

    Returns
    -------
    cl_matrix_cut : ndarray  (n_realizations, n_bins_analysis)
    cl_std_cut    : ndarray  (n_bins_analysis,)
    ell_edges     : ndarray  (n_bins_analysis + 1,)  edges actually used
    """
    n_bins_file = cl_matrix.shape[1]

    # Full edges as stored in the FITS file
    ell_edges_full = np.arange(
        ell_min_file,
        ell_min_file + (n_bins_file + 1) * delta_ell,
        delta_ell,
    )  # shape (n_bins_file + 1,)

    # Find the first bin whose CENTER is >= ell_min
    bin_centers = (ell_edges_full[:-1] + ell_edges_full[1:]) / 2.0
    keep_mask = bin_centers >= ell_min

    if not keep_mask.any():
        raise ValueError(
            f"ell_min={ell_min} excludes ALL bins (max bin center is "
            f"{bin_centers[-1]}). Lower ell_min or check delta_ell.")

    first_keep = np.where(keep_mask)[0][0]

    cl_matrix_cut = cl_matrix[:, first_keep:]
    cl_std_cut    = cl_std[first_keep:]
    ell_edges     = ell_edges_full[first_keep:]  # includes the right edge of last bin

    return cl_matrix_cut, cl_std_cut, ell_edges