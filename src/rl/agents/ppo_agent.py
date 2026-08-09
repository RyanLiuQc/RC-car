# """Proximal Policy Optimization (PPO) Agent: on-policy actor-critic algorithm.
#
# This file implements PPOAgent, inheriting from BaseAgent. It optimizes a stochastic
# policy network using a clipped surrogate loss objective and Generalized Advantage
# Estimation (GAE) to deliver stable continuous vehicle control.
# """

import numpy as np
from src.rl.base_agent import BaseAgent
from src.rl.networks.actor import ActorNetwork
from src.rl.networks.critic import CriticNetwork

import torch
from torch import optim
import torch.nn.functional as F


class PPOAgent(BaseAgent):
    def __init__(
            self, 
            obs_dim: int = 6,
            action_dim: int = 2, 
            actor_lr: float = 1e-4, 
            critic_lr: float = 3e-4) -> None:
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        # TODO: Initialize PolicyNetwork, ValueNetwork, and Adam optimizers.

        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(obs_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select continuous action using PolicyNetwork Gaussian distribution."""
        action_np, log_prob = self.actor.get_action(obs,deterministic=deterministic)
        return action_np

    def train_step(self, trajectory_buffer: dict) -> dict:
        """Perform PPO clipped policy gradient update step."""
        # RECALL
        # trajectory_buffer: dict = {
        #     "obs": obs,
        #     "action": action,
        #     "reward": reward,
        #     "next_obs": next_obs, 
        #     "terminated": terminated
        # }

        obs = trajectory_buffer["obs"]
        action = trajectory_buffer['action']
        reward = trajectory_buffer['reward']
        next_obs = trajectory_buffer['next_obs']
        terminated = trajectory_buffer['terminated']

        # collect set of data
        for i in range(20):
            pass
        
        ratio, A_t, g = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        self.actor_optimizer.zero_grad()
        Loss_actor = min(ratio * A_t, g).mean()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        # Loss_critic = F.mse_loss()
        self.critic_optimizer.step()
        return {}

    def save(self, filepath: str) -> None:
        """Save PPO actor and critic weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)


    def load(self, filepath: str) -> None:
        """Load PPO actor and critic weights from disk."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])