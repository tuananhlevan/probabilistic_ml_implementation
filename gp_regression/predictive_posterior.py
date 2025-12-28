import numpy as np
from gaussian_rbf import gaussian_rbf

def posterior_predictive(x, X_train, y_train, l=1., sigma_f=1., sigma_y=1e-8):
    K = gaussian_rbf(X_train, X_train, l, sigma_f) + np.square(sigma_y) * np.eye(len(X_train))
    K_s = gaussian_rbf(X_train, x, l, sigma_f)
    K_ss = gaussian_rbf(x, x, l, sigma_f)

    mu_s = K_s.T @ np.linalg.inv(K) @ y_train
    cov_s = K_ss - K_s.T @ np.linalg.inv(K) @ K_s
    return mu_s, cov_s