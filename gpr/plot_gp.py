import numpy as np
import matplotlib.pyplot as plt

def plot_gp(mu, cov, X, samples=[]):
    X = X.reshape(-1)
    mu = mu.reshape(-1)

    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.3, label="95% confidence interval")
    plt.plot(X, mu, label='y = sin(x)')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='Prediction {}'.format(i + 1))