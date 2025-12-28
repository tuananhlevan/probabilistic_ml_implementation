import numpy as np

def gaussian_rbf(x1, x2, l=1, sigma_f=1):
    # distance between each rows
    dist_matrix = np.sum(np.square(x1), axis=1).reshape(-1, 1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
    return np.square(sigma_f) * np.exp(-1 / (2 * np.square(l)) * dist_matrix)