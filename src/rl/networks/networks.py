"""
### Transfered to 2 different classes for actor and critic ###
PyTorch neural network architectures: Actor Policy and Critic Value Networks in a single class.

This file defines the neural network building blocks (PolicyNetwork)
that map continuous 6D observation states to control actions (throttle and steering)
and evaluate state values for Actor-Critic algorithms like PPO and SAC.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple

class PolicyNetwork(nn.Module):
    """Policy Network is initializing 2 different Neural Network: 
    Actor Network and Critic Network in the same class. (Actor-Critic Network)
    This can also be done with 2 seperate class
    """

    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden_dim: int = 64) -> None:
        """
        Initialize network layers and log standard deviation parameter.
        """
        super().__init__()
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim

        # Create a shared MLP backbone using nn.Sequential:
        # Backbone (first few layers) is the same for both net (a linear_relu_stack)
        # Linear(obs_dim, hidden_dim) -> ReLU() -> Linear(hidden_dim, hidden_dim) -> ReLU()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Create Actor Mean head: Linear(hidden_dim, action_dim) -> Tanh()
        # apply Linear layer than scale output vector (throttle and steering) to -1 to 1 with tanh
        self.actor_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # the neural network returns a normal distribution for the action to be taken
        # (mean_throttle, std_throttle) and (mean_steering, std_steering)
        # the distribution for the action 
        # is what we are trying to predict based on the observation input!

        # Create Critic Value head: Linear(hidden_dim, 1)
        # It maps the latent feature representation of an observation s 
        # to a single scalar value V(s) (the estimated expected cumulative reward from state s).
        # nn.Parameters lets pytorch now these are trainable weights
        self.critic_value_head = nn.Linear(hidden_dim, 1)

        
        # Define trainable log standard deviation parameter for Gaussian action sampling:
        # self.log_std = nn.Parameter(torch.zeros(action_dim))
        # taking e^log_std to get std will force std to be positive. 
        # we first initialize it to 0,0 
        self.log_std = nn.Parameter(torch.zeros(action_dim))


    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate observation inputs to output normalized controls.
        Returns [throttle, steering] values.

        Runs a single forward pass through the shared backbone.
        Returns:
            action_mean: Tensor of shape (batch, action_dim)
            action_std:  Tensor of shape (action_dim,) or broadcasted
            state_value: Tensor of shape (batch, 1)
    
        """
        # get hidden layers' features (output of the backbone)
        features = self.backbone(observation) 

        # output of the backbone is the input to actor_mean_head
        action_mean = self.actor_mean_head(features) # (mean_throttle, mean_steering)
        action_std = torch.exp(self.log_std)

        state_value = self.critic_value_head(features) # estimated state value

        return action_mean, action_std, state_value

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        For Actor
        Sample action for environment step: returns (clipped_action_numpy, log_prob_tensor, value_tensor).
        """
        # Convert obs numpy array to torch.float32 tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

        # Call self.forward(obs_tensor) to get (mean, std, value)
        mean, std, state_value = self.forward(obs_tensor)

        
        if deterministic:
            # action_tensor = mean
            action = mean
            dist = Normal(mean, 0)
            log_prob = torch.Tensor(1.0)
        else:
            # sample from normal distribution
            dist = Normal(mean, std)
            action = dist.sample()

            # Compute log_prob
            # sum both actions' probability (log throttle_prob and log steering prob)
            # this is the same as multiplying their prob after taking exp of the sum. (independent events)
            log_prob = dist.log_prob(action).sum(dim=-1)



        # Clip action tensor to [-1.0, 1.0] and convert to numpy array
        clipped_action = torch.clamp(action, -1, 1)
        action_np = clipped_action.numpy()

        # Return (action_np, log_prob, value)
        return action_np, log_prob, state_value

    def evaluate_actions(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For Actor
        For On-Policy training (PPO, A2C)
        Used during PPO mini-batch updates: evaluates log_probs, values, and entropy for past trajectory samples.

        The reason evaluate_actions takes obs_batch is that there could be many observations for PPO since
        PPO evaluates action from multidimensional TD error (A_gae: generalized advantage estimation) 
        so it needs many steps with no_grad and no loss computation 
        to collect data before computing loss based on A_gae.

        The vanilla A2CAgent will only have a batch size of 1 since it is 1D TD_error.
        """
        # Call self.forward(obs_batch) -> (mean, std, values)
        mean, std, state_values = self.forward(obs_batch)

        # Create dist = Normal(mean, std)
        dist = Normal(mean, std)

        # Compute log_probs = dist.log_prob(action_batch).sum(dim=-1)
        log_probs = dist.log_prob(action_batch).sum(dim=-1)

        # Compute entropy = dist.entropy().sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Return (log_probs, state_values.squeeze(-1), entropy)
        # squeeze state_values to match actor and critic tensors
        return log_probs, state_values.squeeze(-1), entropy 


    def load_weights(self, path: str):
        pass

# class ValueNetwork(nn.Module):
#     """Critic Value Network estimating state baseline values V(s)."""
#     def __init__(self, obs_dim: int = 6) -> None:
#         super().__init__()
#         self.obs_dim: int = obs_dim
#         # TODO: Implement Critic Value network layers.