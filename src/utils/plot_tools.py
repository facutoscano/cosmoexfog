import numpy as np
import matplotlib.pyplot as plt
from src.utils.stat_tools import compute_empirical_pvalue

def plot_correlation_histograms(param_names, smoothing_angles, data_corrs, sim_corrs, out_path, n_voids):
    """Plot histograms comparing Data vs Sims."""

    fig, axes = plt.subplots(len(param_names), len(smoothing_angles), 
                             figsize=(5 * len(smoothing_angles), 4 * len(param_names)))
    
    if len(param_names) == 1: axes = np.expand_dims(axes, axis=0)
    if len(smoothing_angles) == 1: axes = np.expand_dims(axes, axis=1)

    for i, p_name in enumerate(param_names):
        for j, angle in enumerate(smoothing_angles):
            ax = axes[i, j]
            
            # Datos Reales
            data_vals = data_corrs[angle][p_name]
            data_mean = np.mean(data_vals)
            data_std = np.std(data_vals) if n_voids > 1 else 0.0
            
            # Simulaciones 
            sim_vals_mean = [np.mean(sim_cat_list) for sim_cat_list in sim_corrs[angle][p_name]]
            
            # Cálculo de p-value y sigmas
            p_val, sigma_det = compute_empirical_pvalue(data_mean, sim_vals_mean)
            
            # Ploteo de Simulaciones
            ax.hist(sim_vals_mean, bins=20, histtype='step', color='black', alpha=0.7, label='Sims')
            
            # Ploteo de Datos
            if n_voids > 1:
                ax.axvspan(data_mean - data_std, data_mean + data_std, color='xkcd:watermelon', 
                           alpha=0.2, label=r'Data $1\sigma_{jitter}$')
            ax.axvline(data_mean, color='red', linestyle='dashed', linewidth=2, label=f'Data: {data_mean:.3f}')
            
            # Formato estético con p-value
            ax.set_title(f'{p_name} - {angle}°\np-value: {p_val:.4f} ({sigma_det:.2f} $\\sigma$)')
            if i == len(param_names) - 1:
                ax.set_xlabel('Correlación de Pearson Ponderada')
            ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)