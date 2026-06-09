import numpy as np
from scipy.special import erfinv

def compute_weighted_correlation(map_x, map_y, w_mask):
    """
    Calcula la correlación de Pearson entre dos mapas considerando 
    estrictamente los pesos de la máscara para medias y varianzas.
    """
    valid = w_mask > 0
    x, y, w = map_x[valid], map_y[valid], w_mask[valid]
    
    sum_w = np.sum(w)
    if sum_w == 0: 
        return 0.0
    
    # Medias pesadas
    mean_x = np.sum(w * x) / sum_w
    mean_y = np.sum(w * y) / sum_w
    
    # Varianzas y covarianza pesadas
    var_x = np.sum(w * (x - mean_x)**2) / sum_w
    var_y = np.sum(w * (y - mean_y)**2) / sum_w
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y)) / sum_w
    
    if var_x == 0 or var_y == 0: 
        return 0.0
        
    return cov_xy / np.sqrt(var_x * var_y)

def compute_empirical_pvalue(data_val, sim_vals):
    """
    Calcula el p-value empírico a dos colas y su conversión a sigmas
    asumiendo distribución no necesariamente gaussiana.
    """
    sim_vals = np.array(sim_vals)
    n_sims = len(sim_vals)
    if n_sims == 0: 
        return 1.0, 0.0
        
    # Contamos cuántas simulaciones son más extremas que los datos reales
    mean_sims = np.mean(sim_vals)
    extreme_count = np.sum(np.abs(sim_vals - mean_sims) >= np.abs(data_val - mean_sims))
    
    p_value = (extreme_count + 1) / (n_sims + 1)
    
    # Conversión directa de p-value a Sigmas
    sigma = np.sqrt(2) * erfinv(1 - p_value)
    
    return p_value, sigma