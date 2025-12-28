import numpy as np

def make_moons(n_samples=100, noise=None, random_state=None):
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer moon
    theta_out = np.linspace(0, np.pi, n_samples_out)
    x_out = np.column_stack([
        np.cos(theta_out),
        np.sin(theta_out)
    ])

    # Inner moon
    theta_in = np.linspace(0, np.pi, n_samples_in)
    x_in = np.column_stack([
        1 - np.cos(theta_in),
        1 - np.sin(theta_in) - 0.5
    ])

    X = np.vstack([x_out, x_in])
    y = np.hstack([
        np.zeros(n_samples_out, dtype=int),
        np.ones(n_samples_in, dtype=int)
    ])

    if noise is not None:
        X += rng.normal(scale=noise, size=X.shape)

    return X, y