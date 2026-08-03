# """Advantage Actor-Critic (A2C) Agent: on-policy synchronous actor-critic policy gradient algorithm.
#
# This file defines A2CAgent inheriting from BaseAgent. It uses separate ActorNetwork and CriticNetwork
# instances to select continuous actions, estimate baseline state values V(s), and update policy weights
# using 1-step or multi-step Temporal Difference (TD) advantage estimates.
# """

import numpy as np
import torch
import torch.optim as optim
from src.rl.base_agent import BaseAgent
from src.rl.networks import ActorNetwork, CriticNetwork

class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C) Agent"""

    def __init__(self, obs_dim: int = 6, action_dim: int = 2, actor_lr: float = 1e-4, critic_lr: float = 3e-4, gamma: float = 0.99) -> None:
        """Initialize ActorNetwork, CriticNetwork, Adam optimizers, and discount factor gamma."""
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.gamma: float = gamma

        # Instantiate separate Actor and Critic networks
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.critic = CriticNetwork(obs_dim)

        # Instantiate Adam optimizers with independent learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select continuous action [throttle, steering] for environment step.
        Returns clipped numpy array of shape (2,).
        """
        action_np, _ = self.actor.get_action(obs, deterministic=deterministic)
        return action_np

    def train_step(self, trajectory_buffer: dict) -> dict:
        """
        Perform 1-step or multi-step Advantage Actor-Critic (A2C) gradient update.
        
        Steps:
        1. Query state_value V(s) and next_state_value V(s_next) from CriticNetwork.
        2. Compute Advantage: A_t = R_t + (gamma * V(s_next) * (1 - terminated)) - V(s)
        3. Evaluate new_log_probs and entropy using self.actor.evaluate_actions(obs, action).
        4. Compute Actor Loss: L_actor = - (new_log_probs * Advantage.detach()) - (c2 * entropy)
        5. Compute Critic Loss: L_critic = MSE(V(s), R_t + gamma * V(s_next))
        6. Zero gradients, perform backpropagation, and step optimizers.
        """
        # TODO: Implement A2C gradient updates
        return {}

    def save(self, filepath: str) -> None:
        """Save both Actor and Critic network state dicts into a single checkpoint file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load Actor and Critic network state dicts from a checkpoint file."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])