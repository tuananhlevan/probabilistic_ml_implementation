import torch
import matplotlib.pyplot as plt

def sampling(model, device):
    x = torch.randn(300, 2).to(device)
    n_steps = 8
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
    
    axes[0].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=7, alpha=0.6)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3., 3.)
    axes[0].set_ylim(-3., 3.)
    
    for i in range(n_steps):
        x = model.step(x, time_steps[i], time_steps[i + 1])
        axes[i + 1].scatter(x.cpu().detach()[:, 0], x.cpu().detach()[:, 1], s=7, alpha=0.6)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')
    
    plt.tight_layout()
    plt.show()