import torch
from torch import nn, optim

from model import Flow
from train_function import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10000
LR = 1e-2
MODEL_PATH = "flow.pth"

model = Flow()
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

train(model=model, optimizer=optimizer, loss_fn=loss_fn, epochs=EPOCHS, device=DEVICE, model_path=MODEL_PATH)