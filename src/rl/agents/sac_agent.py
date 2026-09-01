# """Soft Actor-Critic (SAC) Agent: off-policy maximum-entropy actor-critic algorithm.
#
# This file implements SACAgent, inheriting from BaseAgent. It uses an off-policy
# ReplayBuffer, twin Q-networks, and entropy maximization to optimize sample-efficient
# continuous vehicle control.
# """

import numpy as np
from src.rl.base_agent import BaseAgent
from src.rl.replay_buffer import ReplayBuffer
from src.rl.networks import SACActor, SACCritic

import torch
from torch.nn import utils
from torch import optim
import torch.nn.functional as F


class SACAgent(BaseAgent):
    def __init__(
            self, 
            obs_dim: int = 6, 
            action_dim: int = 2, 
            actor_lr: float = 1e-4, 
            critic_lr: float = 3e-4,
            buffer_capacity: int = 100000,
            ) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.actor_lr: float = actor_lr
        self.critic_lr: float = critic_lr

        self.Q1 = SACCritic(obs_dim, action_dim).to(self.device)
        self.Q2 = SACCritic(obs_dim, action_dim).to(self.device)

        # COPY the parameters from Q1 and same thing for Q2
        self.Q1_target = SACCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.Q2_target = SACCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.actor = SACActor(obs_dim=obs_dim, action_dim=action_dim)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim, action_dim=action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.Q_optimizer = torch.optim.Adam(list(self.Q1.parameters())+list(self.Q2.parameters()), lr=critic_lr)
        

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select continuous action using reparameterized PolicyNetwork."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            # maybe need to detach (no grad to make this work before converting to numpy)
            return self.actor(obs)[0].numpy()

        action, log_std = self.actor.sample_action(obs_tensor)

        return action.numpy()

    def train_step(self, trajectory_buffer: dict) -> dict:
        """Perform SAC soft Q-learning update step over mini-batch."""
        # TODO: Implement soft Bellman backup and entropy minimization updates.
        return {}

    def save(self, filepath: str) -> None:
        """Save SAC actor and twin critic weights to disk."""
        pass

    def load(self, filepath: str) -> None:
        """Load SAC actor and twin critic weights from disk."""
        pass