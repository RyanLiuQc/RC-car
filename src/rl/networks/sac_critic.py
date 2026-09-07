"""
Q network. 
Input: obs and action,
Output: value
Loss:   
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Any, List, Tuple


class SACCritic(nn.Module):
    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim+action_dim, out_features=hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            # value 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # (batch_size, obs_dim+action_dim)
        x = torch.concat([obs, action], dim=-1)
        return self.net(x)

