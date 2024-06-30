import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class FNN(nn.Module):
    """Time-Conditioned Feedforward Neural Network"""
    def __init__(self, input_dim=2, time_emb_dim=16):
        super(FNN, self).__init__()

        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)

        self.fc1 = nn.Linear(input_dim + time_emb_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, input_dim)

        # Initialize weights using normal distribution
        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.fc1.weight, mean=0.0, std=1 / math.sqrt(18))
        init.normal_(self.fc2.weight, mean=0.0, std=1 / math.sqrt(128))
        init.normal_(self.fc3.weight, mean=0.0, std=1 / math.sqrt(128))
        init.normal_(self.fc4.weight, mean=0.0, std=1 / math.sqrt(128))

    def forward(self, x, t):
        # Embed the time step
        t_emb = self.time_embedding(t)

        # Concatenate input with time embedding
        x = torch.cat([x, t_emb], dim=1)

        # Pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
