#%% Imports
import numpy as np
import camb

#%% Functions
def get_theory_cls(params_dict, ell_max_req, tau_fid=0.0544):
    """
    Generate the theoretical TT power spectrum D_l [μK²] using CAMB.

    Returns:
        ells      : array of multipole moments
        dls_total : D_l = l(l+1)/(2π) C_l [μK²], including optional point-source term
    """
    As = params_dict['As_e2tau'] * 1e-9 * np.exp(2*tau_fid)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0    = params_dict['H0'],
        ombh2 = params_dict['ombh2'],
        omch2 = params_dict['omch2'],
        omk   = 0,
        tau   = tau_fid,
    )
    pars.InitPower.set_params(As=As, ns=params_dict['ns'], r=0)
    pars.set_for_lmax(ell_max_req + 100, lens_potential_accuracy=0)

    results  = camb.get_results(pars)
    spectrum = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    dls_total = spectrum['total'][:, 0]
    ells      = np.arange(len(dls_total))

    if 'a_ps' in params_dict:
        dls_total = dls_total + params_dict['a_ps'] * ells * (ells + 1) / (3000. * 3001.)

    return ells, dls_total
