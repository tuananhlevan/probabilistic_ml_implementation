# Probabilistic Machine Learning & Generative Models

A collection of PyTorch implementations of fundamental probabilistic models and generative algorithms. The goal of this repository is to reproduce core results from first principles, focusing on mathematical correctness and visualization of internal representations.

## Implementation Gallery

| Model | Key Concept | Visualization |
| :--- | :--- | :--- |
| **[NCSN (Score-Based)](./ncsn)** | Annealed Langevin Dynamics & Score Matching | <img src="./ncsn/assets/langevin_dynamic.gif" width="200px"> <br> *Noise converging to data manifold* |
| **[Gaussian Processes](./gpr)** | Kernel Methods & Uncertainty Quantification | <img src="./gpr/assets/gpr_plot.png" width="200px"> <br> *Posterior confidence intervals* |
| **[VAE](./vae)** | Variational Inference & Latent Manifolds | <img src="./vae/assets/manifold.png" width="200px"> <br> *Continuous latent space interpolation* |

## Implementations List

### 1. Score-Based Generative Models (NCSN)
Implementation of **Noise Conditional Score Networks** (Song & Ermon, 2019).
* **Core Logic:** Estimates the gradient of the log-density $\nabla_x \log p(x)$ (the score) using a U-Net/MLP.
* **Sampling:** Uses **Annealed Langevin Dynamics** to generate samples from noise.
* [View Code & Analysis](./ncsn)

### 2. Variational Autoencoder (VAE)
A deep generative model focusing on the **Evidence Lower Bound (ELBO)**.
* **Core Logic:** Enforces a Gaussian prior on the latent space via KL-Divergence.
* **Key Result:** Demonstrates disentangled, continuous latent representations of MNIST digits.
* [View Code & Analysis](./vae)

### 3. Gaussian Process Regression (GPR)
Non-parametric Bayesian regression.
* **Core Logic:** Exact inference using custom RBF kernels.
* **Key Result:** Visualizes the "collapse" of uncertainty near observed data points.
* [View Code & Analysis](./gpr)