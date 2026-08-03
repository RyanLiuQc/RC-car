"""Critic Network
Network to output state value

"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            # nn.Linear(hidden_dim, 1) # state_value_layer
        )

        self.state_value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.state_value_layer(features) # estimated state value