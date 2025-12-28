import torch

from model import Flow
from visualize import sampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Flow()
model.load_state_dict(torch.load("flow.pth", weights_only=True))
model.to(DEVICE)

sampling(model=model, device=DEVICE)