import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Output Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Output Log Variance

        # Decoder Layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        # Calculate standard deviation
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from standard normal
        eps = torch.randn_like(std)
        # Return sampled z
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)) # Sigmoid for BCE Loss (0-1 range)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) # Flatten if image
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction term: Mean Squared Error (L2 Norm)
        # Sum over all pixels and batch
        MSE = F.mse_loss(recon_x, x.view(-1, 784), reduction='sum')

        # KL Divergence term
        # Analytical result for Standard Normal Prior
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD