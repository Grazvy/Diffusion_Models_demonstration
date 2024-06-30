import torch
import torch.nn as nn


class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_dim=2):
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        batch_size = x.size(0)

        t = t.view(batch_size, 1, 1, 1)
        x_t = torch.cat([x, t], dim=3)
        x_t = x_t.view(batch_size, 3)

        output = self.network(x_t)
        return output.view(batch_size, 1, 1, 2)
