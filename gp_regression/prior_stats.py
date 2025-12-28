import numpy as np
from gaussian_rbf import gaussian_rbf

def prior(X_data):
    mu_p = np.zeros(X_data.shape[0])
    sigma_p = gaussian_rbf(X_data, X_data, 1, 1)
    return mu_p, sigma_p