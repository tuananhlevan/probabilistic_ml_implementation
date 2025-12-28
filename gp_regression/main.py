import numpy as np
import matplotlib.pyplot as plt
from helper_plot import plot_gp
from gaussian_rbf import gaussian_rbf
from prior_stats import prior
from predictive_posterior import posterior_predictive

kernel = gaussian_rbf

X = np.arange(-5, 5, 0.2).reshape(-1, 1)
X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.normal(0, 0.3)

mu_s, cov_s = posterior_predictive(X, X_train, y_train, sigma_y=0.3)
samples = np.random.multivariate_normal(mu_s.reshape(-1), cov_s, 3)

plot_gp(mu_s, cov_s, X, samples)
plt.plot(X_train, y_train, "rx")
plt.show()