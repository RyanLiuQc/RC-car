import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Any, List, Tuple
import torch.nn.functional as F
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_mean = nn.Linear(hidden_dim, action_dim)
            # cannot squeeze early since randomness is introduced later. you need to squeeze action +- epsilon*std
            # nn.Tanh() # squeeze throttle and steering in -1 and 1

        self.log_std = nn.Linear(hidden_dim, 2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        return self.action_mean(x), self.log_std(x)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using reparameterization (rsample) + tanh squashing. action = dist.rsample(Normal(action_mean, std))
        Returns:
            action: squashed action in [-1, 1]
            log_prob: corrected log probability of the squashed action
        """
        # not actually action mean since it is not normalized yet.
        action_mean, log_std = self(obs)
        std = torch.exp(log_std)

        epsilon = Normal(0,1).sample(action_mean.shape).to(action_mean.device)

        # keep action differentiable by seperating action_mean/std and epsilon noise, by reparametrizing
        # instead of defining epsilong, you can also directly use init a Normal() dist
        # then use dist.rsample to get reparametrized action
        u = action_mean+epsilon*std

        # action is btw [-1, 1]
        action = torch.tanh(u) # action is no longer gaussian after squashing...

        # get log probability of this action
        # using chain of variables p(y) = p(x)(dx/dy) apply log
        dist = Normal(action_mean, std)
        log_prob_u = dist.log_prob(u)

        # 1-action**2 == da/du
        log_prob_a = log_prob_u - torch.log(1.0-action.pow(2) + 1e-6).sum() 
        # sum is derived from the jacobian matrix's determinant then taking the log leads to summation.
        # jacobian = (1-action1^2)(1-action2), take log for sum

        return action, log_prob_a