import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import VAE
from data import get_dataloader
from train_function import train

INPUT_DIM = 784
HIDDEN_DIM = 400
LATENT_DIM = 2
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
MODEL_PATH = "VAE.pth"

model = VAE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
train_loader, _ = get_dataloader(batch_size=BATCH_SIZE)

train(model=model, epochs=EPOCHS, optimizer=optimizer, device=DEVICE, train_loader=train_loader, model_path=MODEL_PATH)