#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

#%% Functions
def plot_correlation_histograms(param_names, smoothing_angles, data_corrs, sim_corrs, out_path, n_voids):
    """Plot histograms comparing Data vs Sims.

    Returns:
    - None
    """

    fig, axes = plt.subplots(len(param_names), len(smoothing_angles), 
                             figsize=(5 * len(smoothing_angles), 4 * len(param_names)))
    
    if len(param_names) == 1: axes = np.expand_dims(axes, axis=0)
    if len(smoothing_angles) == 1: axes = np.expand_dims(axes, axis=1)

    for i, p_name in enumerate(param_names):
        for j, angle in enumerate(smoothing_angles):
            ax = axes[i, j]
            
            data_vals = data_corrs[angle][p_name]
            data_mean = np.mean(data_vals)
            data_std = np.std(data_vals) if n_voids > 1 else 0.0
            
            sim_vals = np.mean(sim_corrs[angle][p_name], axis=0)
            sim_mean = np.mean(sim_vals)
            sim_std = np.std(sim_vals)
            
            sigma_det = np.abs(data_mean - sim_mean) / sim_std if sim_std != 0 else 0

            ax.hist(sim_vals, bins=20, histtype='step', color='black', alpha=0.7, label='Sims')
            
            if n_voids > 1:
                ax.axvspan(data_mean - data_std, data_mean + data_std, color='xkcd:watermelon', 
                           alpha=0.2, label=r'Data $1\sigma_{jitter}$')
            ax.axvline(data_mean, color='red', linestyle='dashed', linewidth=2, label=f'Data: {data_mean:.3f}')
            
            ax.set_title(f'{p_name} - {angle}°\nDetección: {sigma_det:.2f} $\sigma$')
            if i == len(param_names) - 1:
                ax.set_xlabel('Correlación')
            ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

