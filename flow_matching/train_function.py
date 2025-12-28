import torch
from torch import Tensor
from tqdm.auto import tqdm

from data import make_moons

def train(model, optimizer, loss_fn, epochs, device, model_path):
    model.train()
    print("Start training...")
    for _ in tqdm(range(1, epochs + 1), desc="Training", leave=False):
        x_1 = Tensor(make_moons(256, noise=0.05)[0]).to(device)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(len(x_1), 1).to(device)
        x_t = ((1 - t) * x_0 + t * x_1).to(device)
        dx_t = (x_1 - x_0).to(device)
        
        optimizer.zero_grad()
        loss_fn(model(x_t, t), dx_t).backward()
        optimizer.step()
    
    print("Finish training!")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in: {model_path}")