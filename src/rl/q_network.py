import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.net(x)