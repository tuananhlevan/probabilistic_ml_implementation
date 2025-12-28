import matplotlib.pyplot as plt
import torch
import numpy as np

from data import MixtureOfGaussians
from time_conditioned_mlp import ScoreNetMLP, GaussianFourierProjection
from sampling import sample_2d_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Sigmas (Geometric Sequence)
sigma_begin = 10.0
sigma_end = 0.01
num_classes = 50 # 50 noise levels
sigmas = torch.tensor(np.geomspace(sigma_begin, sigma_end, num_classes), dtype=torch.float32).to(DEVICE)

torch.serialization.add_safe_globals([ScoreNetMLP])
torch.serialization.add_safe_globals([GaussianFourierProjection])

model = ScoreNetMLP()
model.load_state_dict(torch.load("NCSN.pth", weights_only=True))
model.to(DEVICE)

dataset = MixtureOfGaussians(n_samples=1000)
data = np.array([dataset[i].numpy() for i in range(1000)])
output = sample_2d_data(model, sigmas)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1.scatter(data[:,0], data[:,1], s=5, alpha=0.6)
ax1.set_title("True distribution")
ax2.scatter(output[:, 0], output[:, 1], s=5, alpha=0.6)
ax2.set_title("Recreate")
plt.show()