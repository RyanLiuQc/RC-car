# """Actor Policy Network: mapping continuous 6D state observations to Gaussian action distributions.
#
# This file defines the ActorNetwork PyTorch module. It owns the Probability Math. 
# It evaluates sensory telemetry input states to output Gaussian distribution action parameters
# (mean and standard deviation) for continuous throttle and steering actuation.
# """

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Any, List, Tuple

class ActorNetwork(nn.Module):
    """Actor Policy Network outputting continuous Gaussian action distributions."""

    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden_dim: int = 64) -> None:
        """Initialize backbone MLP layers, actor mean layer, and trainable log_std parameter."""
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(), # using tanh for continuous obs input instead of relu
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # output
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate observation inputs to output normalized controls.
        Returns [throttle, steering] values.

        Runs a single forward pass through the shared backbone.
        Returns:
            action_mean: Tensor of shape (batch, action_dim)
            action_std:  Tensor of shape (action_dim,) or broadcasted
        """
        # get hidden layers' features (output of the backbone)
        features = self.backbone(obs)

        # output of the backbone is the input to actor_mean_head
        action_mean = self.mean_layer(features) # (mean_throttle, mean_steering)
        action_std = torch.exp(self.log_std)

        return action_mean, action_std
    
    def get_action(self, obs_tensor: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample action for environment step during interaction/rollout.
        If deterministic=True, returns action_mean directly without Gaussian noise sampling.
        Returns (action_np, log_prob) where action_np is clamped to [-1.0, 1.0].

        *** Agent classes handle obs_np's conversion to tensor
        """
        # Ensure 2D batch dimension (1, obs_dim) if a 1D single observation is passed
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # single forward pass to get mean and std
        action_mean, action_std = self.forward(obs_tensor)

        # sample from normal distribution
        dist = Normal(action_mean, action_std)

        if deterministic:
            # Set action to mean 
            action = action_mean
        else:
            # Sample from the distribution
            action = dist.sample()

        # Compute log_prob
        # sum both actions' probability (log throttle_prob and log steering prob)
        # this is the same as multiplying their prob after taking exp of the sum. (independent events)
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clip action tensor to [-1.0, 1.0] and convert to numpy array
        clipped_action = torch.clamp(action, -1.0, 1.0)
        action_np = clipped_action.detach().cpu().numpy().squeeze(0)

        return action_np, log_prob
    
    def evaluate_actions(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For On-Policy training (PPO, A2C) actor.
        Used during PPO mini-batch updates: evaluates log_probs, values, and entropy for past trajectory samples.

        The reason evaluate_actions takes obs_batch is that there could be many observations for PPO since
        PPO evaluates action from multidimensional TD error (A_gae: generalized advantage estimation) 
        so it needs many steps with no_grad and no loss computation 
        to collect data before computing loss based on A_gae.

        The vanilla A2CAgent will only have a batch size of 1 since it is 1D TD_error.
        """
        # Call self.forward(obs_batch) -> (mean, std)
        mean, std = self.forward(obs_batch)

        # Create dist = Normal(mean, std)
        dist = Normal(mean, std)

        # Compute log_probs = dist.log_prob(action_batch).sum(dim=-1)
        log_probs = dist.log_prob(action_batch).sum(dim=-1)

        # Compute entropy = dist.entropy().sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy
