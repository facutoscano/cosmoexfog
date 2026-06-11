#%% Imports
import os
import pickle
import numpy as np
from iminuit import Minuit
import concurrent.futures
from src.utils.data_utils import slice_cls, bin_theory_spectrum
from src.utils.camb_utils import get_theory_cls


# ── chi2 factory ─────────────────────────────────────────────────────────────

def create_chi2_function(data_cl, data_std, ell_edges, lmax_req):
    def my_cmb_chi2(om_b, om_cdm, n_s, a_s, h_0, a_ps, tau):
        if not (0.01 < om_b < 0.05 and 0.09 < om_cdm < 0.4 and
                0.8 < n_s < 0.999 and 1.7e-9 < a_s < 2.5e-9 and
                30. < h_0 < 110. and 0.01 < tau < 0.15):
            return 1.e16

        params_dict = {
            'ombh2': om_b, 'omch2': om_cdm, 'ns': n_s,
            'As': a_s, 'H0': h_0, 'tau': tau, 'a_ps': a_ps
        }
        try:
            _, theo_dls = get_theory_cls(params_dict, lmax_req)
        except Exception:
            return 1.e16

        theo_bin = bin_theory_spectrum(theo_dls, ell_edges)
        return np.sum(((theo_bin - data_cl) / data_std) ** 2)

    return my_cmb_chi2


# ── single-sim worker ─────────────────────────────────────────────────────────
# FIX: accepts cl_data_sim (a 1-D row) instead of the full cl_matrix.
# The old signature passed cl_matrix (601×83, ~400 KB) to every task;
# with ~13000 tasks that's >5 GB of IPC traffic. Now each task only carries
# the single row it needs (~664 B each).

def process_single_sim(cl_data_sim, cl_std, ell_edges, cut_info, planck_pr3, sim_idx):
    cut_cl, cut_std, cut_edges = slice_cls(
        cut_info['ell_cut'], cl_data_sim, cl_std, ell_edges, cut_info['low_cut'])

    _nan_row = {p: np.nan for p in ['H0', 'omegamh2', 'ombh2', 'ns', 'tau', 'As_e2tau', 'a_ps']}

    if len(cut_cl) < 5:
        return sim_idx, cut_info['name'], _nan_row

    lmax_req  = int(cut_edges[-1])
    chi2_func = create_chi2_function(cut_cl, cut_std, cut_edges, lmax_req)

    m = Minuit(chi2_func,
               om_b  = planck_pr3['ombh2'],
               om_cdm= planck_pr3['omch2'],
               n_s   = planck_pr3['ns'],
               a_s   = planck_pr3['As'],
               h_0   = planck_pr3['H0'],
               a_ps  = planck_pr3['a_ps'],
               tau   = planck_pr3['tau'])

    m.limits['a_ps'] = (0, 300)
    # Fix a_ps when there's no high-ell constraining power
    if (cut_info['ell_cut'] is not None
            and not cut_info['low_cut']
            and cut_info['ell_cut'] <= 1250):
        m.fixed['a_ps'] = True

    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()

    if not m.valid:
        return sim_idx, cut_info['name'], _nan_row

    v = m.values
    results = {
        'H0':       v['h_0'],
        'omegamh2': v['om_b'] + v['om_cdm'],
        'ombh2':    v['om_b'],
        'omch2':    v['om_cdm'],
        'ns':       v['n_s'],
        'tau':      v['tau'],
        'As_e2tau': v['a_s'] * np.exp(-2 * v['tau']) * 1e9,
        'a_ps':     v['a_ps'],
    }
    return sim_idx, cut_info['name'], results


# ── pipeline orchestrator ─────────────────────────────────────────────────────

def run_iminuit_pipeline(cl_matrix, cl_std, ell_edges, cuts, out_file, config):
    n_sims    = cl_matrix.shape[0]
    n_workers = config['iminuit']['n_workers']
    planck_ref = config['planck_pr3']

    # --- checkpoint resume ---
    results_dict = {}
    if os.path.exists(out_file):
        with open(out_file, 'rb') as f:
            try:
                results_dict = pickle.load(f)
                n_done = sum(len(v) for v in results_dict.get('results', {}).values())
                print(f"  [Resume] Checkpoint found: {n_done} (sim,cut) pairs done.")
            except EOFError:
                print("  [Warning] Corrupt checkpoint — starting from scratch.")

    if 'results' not in results_dict:
        results_dict = {'results': {i: {} for i in range(n_sims)}, 'data_results': {}}

    # --- build task list (skip already done) ---
    tasks = []
    for sim_idx in range(n_sims):
        for cut in cuts:
            if sim_idx == 0:
                if cut['name'] not in results_dict['data_results']:
                    tasks.append((sim_idx, cut))
            else:
                if cut['name'] not in results_dict['results'][sim_idx]:
                    tasks.append((sim_idx, cut))

    if not tasks:
        print("  All work already done. Skipping minimization.")
        return results_dict

    print(f"  Starting {len(tasks)} minimizations with {n_workers} workers...")

    # FIX: pass cl_matrix[sim_idx] (one row) instead of the full matrix.
    # Reduces IPC data from ~5 GB to ~10 MB for a 601-sim job.
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_sim,
                cl_matrix[sim_idx].copy(),   # <-- only the needed row
                cl_std, ell_edges, cut, planck_ref, sim_idx
            ): (sim_idx, cut['name'])
            for sim_idx, cut in tasks
        }

        count = 0
        for future in concurrent.futures.as_completed(futures):
            sim_idx, cut_name, fit_res = future.result()

            row = [fit_res.get(p, np.nan) for p in config['params_names']]
            if sim_idx == 0:
                results_dict['data_results'][cut_name] = row
            else:
                results_dict['results'][sim_idx][cut_name] = row

            count += 1
            if count % 50 == 0:
                with open(out_file, 'wb') as f:
                    pickle.dump(results_dict, f)
                print(f"    [Progress] {count}/{len(tasks)} done. Checkpoint saved.")

    with open(out_file, 'wb') as f:
        pickle.dump(results_dict, f)
    print("  Minimization finished successfully.")
    return results_dict


# ── Cobaya-compatible likelihood factory ──────────────────────────────────────
# FIX: this function was referenced in run_cobaya.py but didn't exist.
# Cobaya injects Cl (D_l in μK² from CAMB) and the sampled param a_ps.

def get_cobaya_likelihood(cut_cl, cut_std, cut_edges):
    """
    Returns a closure suitable as a Cobaya external likelihood.

    Cobaya passes:
      - Cl   : dict, 'tt' key contains D_l = l(l+1)/(2π) C_l [μK²] from CAMB
      - a_ps : float, sampled point-source amplitude

    The closure computes χ²/2 against the binned data.
    """
    lmax_req = int(cut_edges[-1])

    def my_likelihood(Cl, a_ps):
        # D_l is already in μK² — no unit conversion needed
        tt_dl = np.array(Cl.get('tt', np.zeros(lmax_req + 1)))[:lmax_req + 1]
        ls     = np.arange(len(tt_dl))
        # Add residual point-source foreground (same formula as camb_utils)
        tt_dl  = tt_dl + a_ps * ls * (ls + 1) / (3000. * 3001.)
        theo_bin = bin_theory_spectrum(tt_dl, cut_edges)
        return -0.5 * np.sum(((theo_bin - cut_cl) / cut_std) ** 2)

    return my_likelihood
