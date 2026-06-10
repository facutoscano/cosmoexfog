#%% Imports
import os
import pickle
import numpy as np
from iminuit import Minuit
import concurrent.futures
from src.utils.data_utils import slice_cls, bin_theory_spectrum
from src.utils.camb_utils import get_theory_cls

def create_chi2_function(data_cl, data_std, ell_edges, lmax_req):    
    def my_cmb_chi2(om_b, om_cdm, n_s, a_s, h_0, a_ps, tau):
        if not (0.01 < om_b < 0.05 and 0.09 < om_cdm < 0.4 and 
                0.8 < n_s < 0.999 and 1.7e-9 < a_s < 2.5e-9 and 
                30. < h_0 < 110. and 0.01 < tau < 0.15):
            return 1.e16
            
        params_dict = {'ombh2': om_b, 'omch2': om_cdm, 'ns': n_s, 'As': a_s, 'H0': h_0, 'tau': tau, 'a_ps': a_ps}
        
        try:
            ells, theo_dls = get_theory_cls(params_dict, lmax_req)
        except Exception:
            return 1.e16
            
        theo_bin = bin_theory_spectrum(theo_dls, ell_edges)
        
        chi2 = np.sum(((theo_bin - data_cl) / data_std)**2)
        return chi2
    
    return my_cmb_chi2

def process_single_sim(sim_idx, cl_matrix, cl_std, ell_edges, cut_info, planck_pr3):
    cut_cl, cut_std, cut_edges = slice_cls(cut_info['ell_cut'], cl_matrix[sim_idx], cl_std, ell_edges, cut_info['low_cut'])
    
    if len(cut_cl) < 5:
        return sim_idx, cut_info['name'], {p: np.nan for p in ['H0', 'omegamh2', 'ombh2', 'ns', 'tau', 'As_e2tau', 'a_ps']}
        
    lmax_req = int(cut_edges[-1])
    chi2_func = create_chi2_function(cut_cl, cut_std, cut_edges, lmax_req)
    
    m = Minuit(chi2_func, 
               om_b=planck_pr3['ombh2'], 
               om_cdm=planck_pr3['omch2'], 
               n_s=planck_pr3['ns'], 
               a_s=planck_pr3['As'], 
               h_0=planck_pr3['H0'], 
               a_ps=planck_pr3['a_ps'], 
               tau=planck_pr3['tau'])
               
    m.limits['a_ps'] = (0, 300)
    if cut_info['ell_cut'] is not None and not cut_info['low_cut'] and cut_info['ell_cut'] <= 1250:
        m.fixed['a_ps'] = True
    
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    
    if not m.valid:
        return sim_idx, cut_info['name'], {p: np.nan for p in ['H0', 'omegamh2', 'ombh2', 'ns', 'tau', 'As_e2tau', 'a_ps']}
        
    v = m.values
    results = {
        'H0': v['h_0'],
        'omegamh2': v['om_b'] + v['om_cdm'],
        'ombh2': v['om_b'],
        'omch2': v['om_cdm'],
        'ns': v['n_s'],
        'tau': v['tau'],
        'As_e2tau': v['a_s'] * np.exp(-2 * v['tau']) * 1e9,
        'a_ps': v['a_ps']
    }
    return sim_idx, cut_info['name'], results


def run_iminuit_pipeline(cl_matrix, cl_std, ell_edges, cuts, out_file, config):    
    n_sims = cl_matrix.shape[0]
    n_workers = config['iminuit']['n_workers']
    planck_ref = config['planck_pr3']
    
    results_dict = {}
    if os.path.exists(out_file):
        with open(out_file, 'rb') as f:
            try:
                results_dict = pickle.load(f)
                print(f"  [Resume] Checkpoint: {len(results_dict.get('results', {}))} sims...")
            except EOFError:
                print("  [Warning] Starting from scratch...")
                
    if 'results' not in results_dict:
        results_dict = {'results': {i: {} for i in range(n_sims)}, 'data_results': {}}

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
         print("Work already done. Skipping")
         return results_dict

    print(f"Starting {len(tasks)} minimizations in {n_workers} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_single_sim, sim_idx, cl_matrix, cl_std, ell_edges, cut, planck_ref): (sim_idx, cut['name'])
            for sim_idx, cut in tasks
        }
        
        count = 0
        for future in concurrent.futures.as_completed(futures):
            sim_idx, cut_name, fit_res = future.result()
            
            if sim_idx == 0:
                 results_dict['data_results'][cut_name] = [fit_res.get(p, np.nan) for p in config['params_names']]
            else:
                 results_dict['results'][sim_idx][cut_name] = [fit_res.get(p, np.nan) for p in config['params_names']]
            
            count += 1
            if count % 50 == 0:
                 with open(out_file, 'wb') as f:
                     pickle.dump(results_dict, f)
                 print(f"    [Progress] {count}/{len(tasks)} completed. Checkpoint saved.")

    with open(out_file, 'wb') as f:
         pickle.dump(results_dict, f)
    
    print("  Minimization finished with sucess.")
    return results_dict