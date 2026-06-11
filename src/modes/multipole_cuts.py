#%% Imports
import os
import warnings
import numpy as np
from src.utils.io_utils import load_cls_matrix, export_pkl_to_tsv
from src.samplers.run_iminuit import run_iminuit_pipeline
from src.samplers.run_cobaya import run_mcmc
from src.modes.plots import plot_parameter_grid, plot_shift_grid
from src.utils.data_utils import build_ell_edges_and_slice_data

warnings.filterwarnings('ignore')

#%% Functions
def generate_cuts_list(cfg):
    cuts = [{'name': 'full_spectrum', 'ell_cut': None, 'low_cut': False}]
    for lmin in cfg.get('lmin_cuts', []):
        cuts.append({'name': f'lmin_{lmin}', 'ell_cut': lmin, 'low_cut': True})
    for lmax in cfg.get('lmax_cuts', []):
        cuts.append({'name': f'lmax_{lmax}', 'ell_cut': lmax, 'low_cut': False})
    return cuts


def run(args, config):
    print("\n=== Starting Pipeline: Multipole Cuts ===")

    cfg      = config['multipole_cuts']
    paths    = config['paths']
    sampler  = getattr(args, 'sampler', 'iminuit')
    delta_ell = cfg['delta_ell']
    cuts      = generate_cuts_list(cfg)

    for mask_suffix in cfg.get('mask_variants', ['']):
        dataset_name = cfg['cls_name'] + mask_suffix
        label = mask_suffix if mask_suffix else '(baseline)'
        print(f"\n>> Using mask: {label}")

        cls_path = os.path.join(paths['cls_folder'], f"{dataset_name}.fits")
        try:
            cl_matrix, cl_std, _ = load_cls_matrix(cls_path)
            print(f"  [+] Data loaded: {cl_matrix.shape[0]-1} simulations + Data. "
                  f"Bins per spectrum: {cl_matrix.shape[1]}")
        except FileNotFoundError:
            print(f"  [!] File not found: {cls_path}. Skipping...")
            continue

        n_bins_file  = cl_matrix.shape[1]
        ell_min_file = cfg.get('ell_min_file', 2)
        ell_min      = cfg.get('ell_min', 32)

        
        cl_matrix, cl_std, ell_edges = build_ell_edges_and_slice_data( cl_matrix, cl_std, ell_min_file, ell_min, delta_ell)

        print(f"  [+] FITS bins: {n_bins_file} | analysis bins: {cl_matrix.shape[1]} "
              f"| ell range: {ell_edges[0]}..{ell_edges[-1]}")
        

        out_dir   = os.path.join(paths['output_folder'], 'multipole_cuts', dataset_name)
        plots_dir = os.path.join(paths['plots_folder'],  'multipole_cuts', dataset_name)
        os.makedirs(out_dir,   exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        results_file = ""

        if sampler == 'iminuit':
            out_pkl      = os.path.join(out_dir, f"Minimization_results_{dataset_name}.pkl")
            run_iminuit_pipeline(cl_matrix, cl_std, ell_edges, cuts, out_pkl, cfg)
            results_file = os.path.join(out_dir, f"Params_Iminuit_{dataset_name}.txt")
            export_pkl_to_tsv(out_pkl, results_file, cfg['params_names'])
            print(f"  [+] Results in: {results_file}")

        elif sampler == 'cobaya':
            print("[>] Starting Cobaya chains MCMC...")
            sim_indices = cfg.get('sim_indices') or range(cl_matrix.shape[0])
            for sim_idx in sim_indices:
                data_cl   = cl_matrix[sim_idx]
                sim_label = "data" if sim_idx == 0 else f"sim{sim_idx:04d}"
                for cut_info in cuts:
                    out_prefix = os.path.join(out_dir, f"mcmc_{sim_label}_{cut_info['name']}")
                    run_mcmc(cut_info, data_cl, cl_std, ell_edges,
                             cfg['planck_pr3'], out_prefix, config, rank=0)
            print("  [+] Cobaya MCMC finished for this dataset")

        if results_file and os.path.exists(results_file):
            print("  [>] Generating plots...")
            plot_parameter_grid(results_file, cfg['params_names'], plots_dir, cfg['planck_pr3'])
            plot_shift_grid(    results_file, cfg['params_names'], plots_dir, cfg['planck_pr3'])

    print("\n======================================================")
    print("PIPELINE MULTIPOLE CUTS — FINISHED.")
    print("======================================================")
