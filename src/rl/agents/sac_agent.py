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
            # n_epochs: int = 10, # we don't go through ALL element of dataset... 
            # instead, we just pick randomly sample 32 batches of 64-step (batch_size) (still 2048 steps processed)

            batch_size: int = 64,
            entropy_coef: float = 0.01,
            ) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

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

        # since q1 and q2 weights come from different branches of the computational graph, 
        # they won't affect each other during backward computation.
        self.Q_optimizer = torch.optim.Adam(list(self.Q1.parameters())+list(self.Q2.parameters()), lr=critic_lr)

        # self._count = 0 # keep track of training steps done
        

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select continuous action using reparameterized PolicyNetwork."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            # maybe need to detach (no grad to make this work before converting to numpy)
            return self.actor(obs)[0].numpy()

        action, log_std = self.actor.sample_action(obs_tensor)

        return action.numpy()

    def train_step(self, trajectory_buffer: dict, step: int) -> dict:
        """Perform SAC soft Q-learning update step over mini-batch.
        Input: trajectory_buffer
        Output: metrics = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies)
        }
        """
        # TODO: Implement soft Bellman backup and entropy minimization updates.
        # RECALL
        # trajectory_buffer: dict = {
        #     "obs": obs,
        #     "action": action,
        #     "reward": reward,
        #     "next_obs": next_obs, 
        #     "terminated": terminated,
        #     "done": done # where done = terminated or truncated
        # }
        
        # self._count += 1

        # retrieve data
        obs: np.ndarray = trajectory_buffer["obs"] 
        action: np.ndarray = trajectory_buffer["action"]
        reward: float = trajectory_buffer["reward"]
        next_obs: np.ndarray = trajectory_buffer["next_obs"]
        terminated: bool = trajectory_buffer["done"] # just a choice of terminology. I want to treat any episode ending as a termination in the replay_buffer

        # add to replay buffer
        self.replay_buffer.add(obs,action,reward,next_obs,terminated)

        
        if step < 5000:
            return {}

        # sample and update at every step above 5000 (add step argument to train_step)

        return {}

    def save(self, filepath: str) -> None:
        """Save SAC actor and twin critic weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.Q1.state_dict(),
            'q2_state_dict': self.Q2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.Q_optimizer.state_dict()
        },
        filepath
        )

    def load(self, filepath: str) -> None:
        """Load SAC actor and twin critic weights from disk."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.Q1.load_state_dict(checkpoint['q1_state_dict'])
        self.Q2.load_state_dict(checkpoint['q2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])
