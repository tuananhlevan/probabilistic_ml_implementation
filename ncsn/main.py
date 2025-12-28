import torch
import numpy as np

from data import MixtureOfGaussians
from time_conditioned_mlp import ScoreNetMLP
from sampling import sample_2d_data
from comparison import comparison
from langevin_dynamic import langevin_dynamic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Sigmas (Geometric Sequence)
sigma_begin = 10.0
sigma_end = 0.01
num_classes = 50 # 50 noise levels
sigmas = torch.tensor(np.geomspace(sigma_begin, sigma_end, num_classes), dtype=torch.float32).to(DEVICE)

model = ScoreNetMLP()
model.load_state_dict(torch.load("NCSN.pth", weights_only=True))
model.to(DEVICE)

dataset = MixtureOfGaussians(n_samples=1000)
data = np.array([dataset[i].numpy() for i in range(1000)])
output, history = sample_2d_data(model, sigmas)

comparison(data, output)
langevin_dynamic(history)

