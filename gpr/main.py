import numpy as np
import matplotlib.pyplot as plt
from plot_gp import plot_gp
from gaussian_rbf import gaussian_rbf
from prior_stats import prior
from predictive_posterior import predictive_posterior

np.random.seed(40)

kernel = gaussian_rbf

X = np.arange(-5, 5, 0.2).reshape(-1, 1)
X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.normal(0, 0.)

mu_s, cov_s = predictive_posterior(X, X_train, y_train, sigma_y=0.)
samples = np.random.multivariate_normal(mu_s.reshape(-1), cov_s, 3)

plt.figure(figsize=(8, 6))
plot_gp(mu_s, cov_s, X, samples)
plt.plot(X_train, y_train, "rx", label="Training data")
plt.title(r"GPR Posterior with RBF Kernel: $k(x, x') = \sigma^2 \exp\left(-\frac{(x-x')^2}{2l^2}\right)$")
plt.legend()
plt.show()
# plt.savefig("GPR Posterior with RBF Kernel", dpi=600)