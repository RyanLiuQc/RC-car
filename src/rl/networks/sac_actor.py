import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Any, List, Tuple
import torch.nn.functional as F
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden_dim: int = 64) -> None:
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_mean = nn.Linear(hidden_dim, action_dim)
            # cannot squeeze early since randomness is introduced later. you need to squeeze action +- epsilon*std
            # nn.Tanh() # squeeze throttle and steering in -1 and 1

        self.std = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        return self.action_mean(x), self.std(x)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using reparameterization (rsample) + tanh squashing. action = dist.rsample(Normal(action_mean, std))
        Returns:
            action: squashed action in [-1, 1]
            log_prob: corrected log probability of the squashed action
        """
        action_mean, std = self.forward(obs)

        epsilon = Normal(0,1).sample(action_mean.shape).to(action_mean.device)

        # keep action differentiable by seperating action_mean/std and epsilon noise, by reparametrizing
        u = action_mean+epsilon*std

        action = torch.tanh(u) # action is no longer gaussian after squashing...

        # get log probability of this action
        # using chain of variables p(y) = p(x)(dx/dy) apply log
        dist = Normal(action_mean, std)
        log_prob_u = dist.log_prob(u)

        # 1-action**2 == da/du
        log_prob_a = log_prob_u - dist.log_prob(1-action.pow(2) + 1e-6)

        return action, log_prob_a
    

