#%% Imports
import os
import numpy as np
from cobaya.run import run
from getdist import loadMCSamples
from src.utils.data_utils import slice_cls
from src.samplers.run_iminuit import get_cobaya_likelihood


#%% Functions

def build_cobaya_info(cut_cl, cut_std, cut_edges, ref_dict_PR3, out_prefix, config, a_ps_dynamic, param_limits, tau_fid):
    """Build the Cobaya info dict for a single (data, cut) run."""

    info = {
        "params":
        {
            "ombh2":    {"prior": param_limits['ombh2'],
                         "ref": ref_dict_PR3['ombh2'],"proposal": 0.0001, "latex": r"\omega_b"},
            "omch2":    {"prior": param_limits['omch2'],
                         "ref": ref_dict_PR3['omch2'],
                         "proposal": 0.001, "latex": r"\omega_c"},
            "ns":       {"prior": param_limits['ns'],
                         "ref": ref_dict_PR3['ns'],
                         "proposal": 0.005, "latex": r"n_s"},
            "As_e2tau": {"prior": param_limits['As_e2tau'],
                         "ref": ref_dict_PR3['As_e2tau'],
                         "proposal": 0.01,
                         "latex": r"10^9 A_s e^{-2\tau}"},
            "As":       {"value": f"lambda As_e2tau:As_e2tau * 1e-9 * np.exp(2 * {tau_fid})",
                         "latex": r"A_s"},
            "tau":      {"value": tau_fid,
                         "latex": r"\tau"},
            "H0":       {"prior": param_limits['H0'],
                         "ref": ref_dict_PR3['H0'],
                         "proposal": 0.5, "latex":"H_0"},
            "a_ps":     a_ps_dynamic,
            "omegamh2": {"derived": "lambda ombh2, omch2: ombh2 + omch2",
                         "latex": r"\omega_m"},
},
        "likelihood": {
            "my_cmb_like": {
                "external": get_cobaya_likelihood(cut_cl, cut_std, cut_edges),
                "requires":  {"Cl": {"tt": int(cut_edges[-1])}},
            }
        },
        "theory": {"camb": {"extra_args": {
            "lens_potential_accuracy": 0,
            "num_massive_neutrinos": 1,
        }}},
        "sampler": {"mcmc": {
            "Rminus1_stop": config['mcmc_setup']['Rminus1_stop'],
            "max_tries":    config['mcmc_setup']['max_tries'],
        }},
        "output": out_prefix,
        "resume": True,
    }
    return info


def run_mcmc(cut_info, data_cl, data_std, ell_edges,
             ref_dict_PR3, out_prefix, config, rank=0):
    """Run Cobaya MCMC for one (data, cut) and return mean/std stats."""

    cut_cl, cut_std, cut_edges = slice_cls(
        cut_info['ell_cut'], data_cl, data_std, ell_edges, cut_info['low_cut'])
    if len(cut_cl) < 5:
        return None

    # a_ps: free above ell_max=1250 (point sources matter), fixed below
    # FIX: Cobaya requires a dict — {"value": X} to fix, or a prior dict to sample.
    # The old code passed a raw float (ref_dict_PR3['a_ps']) which crashed Cobaya.
    if (cut_info['ell_cut'] is not None
            and not cut_info['low_cut']
            and cut_info['ell_cut'] <= 1250):
        a_ps_setup = {"value": ref_dict_PR3['a_ps']}            # fixed
    else:
        a_ps_setup = {"prior": {"min": 0.0, "max": 300.0},      # free
                      "ref": ref_dict_PR3['a_ps'], "proposal": 2.0}

    info = build_cobaya_info(cut_cl, cut_std, cut_edges,
                             ref_dict_PR3, out_prefix, config, a_ps_setup)

    # Run only if chain doesn't already exist
    if not os.path.exists(out_prefix + ".1.txt"):
        if rank == 0:
            print(f"  -> Running MCMC for {cut_info['name']}...")
        run(info)
    else:
        if rank == 0:
            print(f"  -> Chain found for {cut_info['name']}. Skipping.")

    # Extract stats (rank-0 only in MPI runs)
    stats = {}
    if rank == 0:
        samples = loadMCSamples(out_prefix, settings={'ignore_rows': 0.3})
        for param in ['H0', 'ombh2', 'omch2', 'ns', 'As_e2tau', 'tau', 'omegamh2']:
            try:
                idx  = samples.index[param]
                stats[param] = {
                    'mean': samples.getMeans()[idx],
                    'std':  np.sqrt(samples.getVars()[idx]),
                }
            except KeyError:
                continue
    return stats
