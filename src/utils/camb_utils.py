#%% Imports
import numpy as np
import camb
from camb import model

#%% Functions
def get_theory_cls(params_dict, ell_max_req):
    """
    Generate the theoretical power spectrum D_l using CAMB and the TT spectrum.

    Returns:
    - ell: Multipole moments.
    - dls_total: Theoretical power spectrum.
    """

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=params_dict['H0'], 
        ombh2=params_dict['ombh2'], 
        omch2=params_dict['omch2'], 
        omk=0, 
        tau=params_dict.get('tau', 0.0544) # Default 
    )
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'], r=0)
    
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    pars.NonLinear = model.NonLinear_none
    pars.set_for_lmax(ell_max_req + 100, lens_potential_accuracy=0)
    
    results = camb.get_results(pars)
    spectrum = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    # Theoretical D_l (Only Temperature)
    dls_total = spectrum['total'][:, 0] 
    ells = np.arange(len(dls_total))
    
    # Residual foreground (a_ps)
    if 'a_ps' in params_dict:
        dls_total += params_dict['a_ps'] * ells * (ells + 1) / (3000. * 3001.)
        
    return ells, dls_total