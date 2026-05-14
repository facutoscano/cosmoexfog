#%% Imports
import numpy as np

#%% Functions
def compute_weighted_correlation(map_x, map_y, w_mask):
    """
    Calculate the weighted Pearson correlation between two maps.

    Returns:
    - corr: Weighted Pearson correlation coefficient.
    """
    # Global Means - Compute unweighted means
    mean_x = np.mean(map_x)
    delta_x = map_x - mean_x
    sigma_x = np.sqrt(np.mean(delta_x**2))

    mean_y = np.mean(map_y)
    delta_y = map_y - mean_y
    sigma_y = np.sqrt(np.mean(delta_y**2))

    # Covariance - Weighted cross-covariance
    if sigma_x == 0 or sigma_y == 0:
        return 0.0
        
    corr = np.mean(delta_x * delta_y * w_mask) / (sigma_x * sigma_y * np.mean(w_mask))
    
    return corr