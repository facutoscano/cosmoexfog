#%% Imports
import os
import pandas as pd
import healpy as hp
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
import numpy as np
import pickle

#%% Functions
def get_voids_file_list(voids_folder, voids_type, n_voids):
    """Return a list of voids file paths."""

    if voids_type not in ['full_sample', 'volume_complete']:
        raise ValueError("Invalid voids_type. Choose 'full_sample' or 'volume_complete'.")

    voids_files = []
    prefix = 'voids_z0.03_' if voids_type == 'full_sample' else 'voids_z0.03_vc_'
    for i in range(n_voids):
        voids_files.append(os.path.join(voids_folder, voids_type, f'{prefix}{(i+1):03d}.dat'))
    return voids_files

def load_parameter_maps(cmb_folder, pref, cut, nside):
    """Load, downgrade and scale the real H0 and Omch2 parameter maps.

    Returns:
        dict: A dictionary containing the parameter maps.
    """
    
    parameters_maps = {}
    
    # H0
    h0_path = os.path.join(cmb_folder, f'ParameterH0{pref}_30degOEmask_2048Cls_lmin32_lmax2000_dl30_JKs_1_12288.fits')
    h0_map = hp.read_map(h0_path)
    h0_map = hp.ud_grade(h0_map, nside_out=nside)
    h0_map = (h0_map + 100) * cosmo.H0.value / 100.0
    parameters_maps[f'H0_{cut}'] = h0_map
    
    # Omch2
    om_path = os.path.join(cmb_folder, f'ParameterOmch2{pref}_30degOEmask_2048Cls_lmin32_lmax2000_dl30_JKs_1_12288.fits')
    om_map = hp.read_map(om_path)
    om_map = hp.ud_grade(om_map, nside_out=nside)
    om_map = (om_map + 100) * cosmo.Odm0 * cosmo.h**2 / 100.0
    parameters_maps[f'Omch2_{cut}'] = om_map
    
    return parameters_maps

def get_sim_path(cmb_folder, param_cut, p_name, sim_idx):
    """Construct the correct path to read a simulation given its configuration."""
    base_param = 'H0' if 'H0' in p_name else 'Omch2'
    
    if param_cut == 'normal':
        sim_dir = os.path.join(cmb_folder, 'Params_300sims', base_param)
        return os.path.join(sim_dir, f'{base_param}_param_sim_{sim_idx}.fits')
    else:
        sim_dir = os.path.join(cmb_folder, 'Params_300sims_ell23', base_param)
        ell_val = '2' if param_cut == 'no_ell2' else '3'
        param_str = 'h0' if base_param == 'H0' else 'omch2'
        return os.path.join(sim_dir, f'simulation{sim_idx}_rm_l0{ell_val}_{param_str}.fits')
    
def load_cls_matrix(cls_path):
    """
    Carga un archivo FITS de Cls.
    Retorna la matriz completa y la desviación estándar de las simulaciones.
    """
    with fits.open(cls_path) as hdul:
        data = hdul[0].data
        
    if data.ndim == 2:
        cl_matrix = data
        cl_std = np.std(data[1:], axis=0) 
    else:
        cl_matrix = np.array([data])
        cl_std = np.ones_like(data)
        
    return cl_matrix, cl_std, None

def export_pkl_to_tsv(pkl_path, tsv_path, params_names):
    if not os.path.exists(pkl_path):
        print(f"  [!] File {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    data_results = data.get('data_results', {})
    sim_results = data.get('results', {})

    header = ['sim', 'cut'] + params_names

    with open(tsv_path, 'w') as f:
        f.write('\t'.join(header) + '\n')

        for cut_name, param_values in data_results.items():
            row = ['0', cut_name] + [f"{val:.6e}" if (val is not None and not np.isnan(val)) else "NaN" for val in param_values]
            f.write('\t'.join(row) + '\n')

        for sim_idx, cuts_dict in sim_results.items():
            if sim_idx == 0: continue
            for cut_name, param_values in cuts_dict.items():
                row = [str(sim_idx), cut_name] + [f"{val:.6e}" if (val is not None and not np.isnan(val)) else "NaN" for val in param_values]
                f.write('\t'.join(row) + '\n')