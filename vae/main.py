import torch

from model import VAE
from plot_latent_manifold import plot_latent_manifold
from sampling import sampling

INPUT_DIM = 784
HIDDEN_DIM = 400
LATENT_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
model.load_state_dict(torch.load("VAE.pth", weights_only=True))
model.to(DEVICE)
plot_latent_manifold(model=model, device=DEVICE)
sampling(model=model, device=DEVICE, grid=4)
