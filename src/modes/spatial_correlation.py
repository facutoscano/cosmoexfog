#%% Imports
import os
import numpy as np
import healpy as hp
import pandas as pd

from src.utils.map_tools import project_voids_to_map, smooth_map
from src.utils.stat_tools import compute_weighted_correlation
from src.utils.plot_tools import plot_correlation_histograms
from src.utils.io_utils import get_voids_file_list, load_parameter_maps, get_sim_path

def run(args, config):
    print("=== Starting Spatial Correlation Pipeline ===")

    paths = config['paths']
    sc_conf = config['spatial_correlation']
    
    if not sc_conf.get('smoothing_angles'):
        raise ValueError("CRITICAL ERROR: 'smoothing_angles' is None or empty. A smoothing angle is strictly required.")
    
    cmb_folder = paths['cmb_folder']
    maps_folder = os.path.join(paths['output_folder'], 'MAPS_SpatialCorr')
    plots_folder = os.path.join(paths['plots_folder'], 'SpatialCorr')
    os.makedirs(maps_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    param_cut_map = {'base': 'normal', 'l02': 'no_ell2', 'l03': 'no_ell3'}
    param_pref_map = {'normal': '', 'no_ell2': '_removingell0-2', 'no_ell3': '_removingell0-3'}
    
    # Load Voids File List
    voids_files = get_voids_file_list(paths['voids_folder'], sc_conf['voids_type'], sc_conf['n_voids'])

    # Main Loop
    for config_run in sc_conf['configs_to_run']:
        print(f"\n--- Processing Configuration: {config_run.upper()} ---")

        param_cut = param_cut_map[config_run]
        pref = param_pref_map[param_cut]
        param_names = [f'H0_{param_cut}', f'Omch2_{param_cut}']

        data_corrs = {ang: {p: [] for p in param_names} for ang in sc_conf['smoothing_angles']}
        sim_corrs = {ang: {p: [] for p in param_names} for ang in sc_conf['smoothing_angles']}

        # 2. Load Real Parameter Maps
        print('  > Loading parameter maps...')
        parameters_maps = load_parameter_maps(cmb_folder, pref, param_cut, sc_conf['nside'])
        
        for angle in sc_conf['smoothing_angles']:
            print(f"\n  > Evaluating FWHM = {angle}°")

            cmask_path = os.path.join(cmb_folder, f"Common_mask_Temperature_Smoothed_nside{sc_conf['nside']}_{int(angle)}deg.fits")
            
            if not os.path.exists(cmask_path):
                print(f"    [Generating] Smoothed mask at {angle}° not found. Creating...")
                base_mask = hp.read_map(os.path.join(cmb_folder, 'Common_mask_Temperature_2048.fits'))
                base_mask = hp.ud_grade(base_mask, nside_out=sc_conf['nside'])
                base_mask[base_mask < 0.5] = 0
                base_mask[base_mask >= 0.5] = 1
                
                smoothed_mask = smooth_map(base_mask, angle)
                hp.write_map(cmask_path, smoothed_mask, overwrite=True)
            else:
                smoothed_mask = hp.read_map(cmask_path)

            for idx_cat in range(sc_conf['n_voids']):
                base_name = f"rmin{sc_conf['r_min']}_rmax{sc_conf['r_max']}_nside{sc_conf['nside']}_cat{idx_cat:03d}_{sc_conf['frac_rvoid']}Rvoid_smooth{int(angle)}deg.fits"
                map_path = os.path.join(maps_folder, f"cmb_voids_{base_name}")
                
                if not os.path.exists(map_path):
                    print(f"    [Generating] Voids Map {base_name} not found. Processing and smoothing...")

                    voids_df = pd.read_csv(voids_files[idx_cat], sep='\s+', names=['R', 'l', 'b', 'redshift', 'x', 'y', 'z', 'delta1', 'delta23', 'flag', 'delta_LOS'])
                    voids_final = voids_df[(voids_df['R'] >= sc_conf['r_min']) & (voids_df['R'] <= sc_conf['r_max']) & (voids_df['delta_LOS'] <= sc_conf['deltalos_max'])]
                    
                    cmb_base_map = hp.read_map(os.path.join(cmb_folder, 'SMICA_2048_PR3.fits')) * 1e6
                    cmb_base_map = hp.ud_grade(cmb_base_map, nside_out=sc_conf['nside'])
                    
                    _, valid_mask, select_indices = project_voids_to_map(voids_final, sc_conf['nside'], smoothed_mask, sc_conf['frac_rvoid'])
                    
                    cmb_safe = np.zeros(hp.nside2npix(sc_conf['nside']))
                    cmb_safe[select_indices] = cmb_base_map[select_indices]
                    
                    smoothed_cmb = smooth_map(cmb_safe, angle)
                    smoothed_weights = smooth_map(valid_mask, angle)
                    
                    cmb_voids_smoothed = np.full(hp.nside2npix(sc_conf['nside']), np.nan)
                    valid_smooth_pixels = smoothed_weights > 1e-5
                    cmb_voids_smoothed[valid_smooth_pixels] = smoothed_cmb[valid_smooth_pixels] / smoothed_weights[valid_smooth_pixels]
                    
                    hp.write_map(map_path, cmb_voids_smoothed, overwrite=True)
                else:
                    cmb_voids_smoothed = hp.read_map(map_path)
                
                valid_idx = ~np.isnan(cmb_voids_smoothed)
                
                map_x = cmb_voids_smoothed[valid_idx]
                w_mask = smoothed_mask[valid_idx]

                # ----- DATA CORRELATION -----
                for p_name in param_names:
                    map_y = parameters_maps[p_name][valid_idx]
                    corr = compute_weighted_correlation(map_x, map_y, w_mask)
                    data_corrs[angle][p_name].append(corr)

                # ----- SIMULATIONS BLOCK -----
                n_sims = sc_conf.get('n_sims')
                if n_sims is not None and idx_cat == 0:
                    print(f'    > Running {n_sims} simulations for {angle}°...')

                    for p_name in param_names:
                        cat_sims = []
                        for s in range(n_sims):
                            sim_file = get_sim_path(cmb_folder, param_cut, p_name, s)
                            sim_map = hp.read_map(sim_file)
                            sim_map = hp.ud_grade(sim_map, nside_out=sc_conf['nside'])
                            
                            map_y_sim = sim_map[valid_idx]
                            
                            corr_sim = compute_weighted_correlation(map_x, map_y_sim, w_mask)
                            cat_sims.append(corr_sim)
                            
                        sim_corrs[angle][p_name].append(cat_sims)

        # 3. Final Plotting
        out_name = f"rmin{sc_conf['r_min']}_rmax{sc_conf['r_max']}_nside{sc_conf['nside']}_Ncat{sc_conf['n_voids']}_{config_run}.pdf"
        out_path = os.path.join(plots_folder, f"Cross_correlation_{out_name}")
        
        print(f"  > Generating plots for {config_run.upper()}...")
        plot_correlation_histograms(param_names, sc_conf['smoothing_angles'], 
                                    data_corrs, sim_corrs, out_path, sc_conf['n_voids'])
        
        print(f"Plot successfully saved to: {out_path}")
        
    print("\n======================================================")
    print("SPATIAL CORRELATION PIPELINE FINISHED SUCCESSFULLY.")
    print("======================================================")