import torch
from torch import optim
import numpy as np

from time_conditioned_mlp import ScoreNetMLP
from data import MixtureOfGaussians
from train_function import train

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ScoreNetMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = MixtureOfGaussians(n_samples=10000)
modelPath = "NCSN.pth"

# Define Sigmas (Geometric Sequence)
sigma_begin = 10.0
sigma_end = 0.01
num_classes = 50 # 50 noise levels
sigmas = torch.tensor(np.geomspace(sigma_begin, sigma_end, num_classes), dtype=torch.float32).to(device)

train(model=model, optimizer=optimizer, sigmas=sigmas, num_classes=num_classes, device=device, dset=dataset, modelPath=modelPath)