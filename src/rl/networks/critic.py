# """Critic Value Network: mapping continuous 6D state observations to scalar state values.
#
# This file defines the CriticNetwork PyTorch module. It maps sensory telemetry observation states
# to an estimated expected cumulative reward scalar baseline V(s), used by Actor-Critic algorithms
# (A2C, PPO) to evaluate action advantages.
# """

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple

class CriticNetwork(nn.Module):
    """Critic Value Network estimating scalar state baseline values V(s)."""

    def __init__(self, obs_dim: int = 6, hidden_dim: int = 64) -> None:
        """Initialize backbone MLP layers and state value linear layer."""
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
        """Forward pass: returns scalar state value V(s) tensor of shape (batch, 1)."""
        features = self.backbone(obs)
        return self.state_value_layer(features) # estimated state value