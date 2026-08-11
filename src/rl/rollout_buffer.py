"""On-Policy Rollout Buffer: trajectory memory and Generalized Advantage Estimation (GAE).

This file defines the RolloutBuffer data structure for on-policy reinforcement learning
algorithms (such as Proximal Policy Optimization - PPO). It accumulates fixed-horizon (N-step)
trajectory sequences of observations, actions, log probabilities under the current policy,
scalar rewards, baseline state values V(s), and termination flags. It computes Generalized
Advantage Estimation (GAE-lambda) and Temporal Difference returns, yielding randomized mini-batches
for multi-epoch stochastic gradient descent policy updates.
"""

import numpy as np
import torch
from typing import Generator, Dict, Any


class RolloutBuffer:
    """On-policy Rollout Buffer for trajectory memory and GAE computation."""

    def __init__(
        self,
        buffer_size: int = 2048,
        obs_dim: int = 6,
        action_dim: int = 2,
        device: str = "cpu"
    ) -> None:
        """Initialize numpy storage arrays and tracking pointers."""
        self.buffer_size: int = buffer_size
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device = torch.device(device)
        
        self.reset()

    def reset(self) -> None:
        """Reset buffer pointers and zero out trajectory memory arrays."""
        self.ptr = 0
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32) # state values

        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
    

    def is_full(self) -> bool:
        """Check if the rollout buffer capacity has been reached."""
        return self.ptr >= self.buffer_size

    def add(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        log_prob: float, 
        reward: float, 
        value: float, 
        done: bool
    ) -> None:
        """Store a single transition step into rollout memory."""
        if self.ptr >= self.buffer_size:
            return
        
        # store
        p = self.ptr
        self.observations[p] = obs
        self.actions[p] = action
        self.log_probs[p] = log_prob
        self.rewards[p] = reward
        self.values[p] = value
        self.dones[p] = done

        self.ptr += 1


    def compute_gae(
        self, 
        last_value: float, # 
        last_done: bool, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and TD returns backward in time."""

        last_gae = 0.0

        # compute advantage for every step starting at N-1 to 0
        # compute 1D td:
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                # check if we finish an episode at last step
                next_non_terminal = 1.0 - float(last_done)

                # get next state value 
                next_value = last_value
            else:
                # check if we finish an episode 
                next_non_terminal = 1.0 - self.dones[t+1]

                next_value = self.values[t+1]

            # Compute 1-step TD Error delta_t, scale to 0 if terminated to keep only final reward
            # avoid advantage bleeding between episode we do not want the next state's resetted value
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]

            # compute estimate of advantage 
            # A_t approx = Q_hat - V(s), where Q_hat = r_t + yV(s_t+1)
            # again, we mask the next gae value if we are at the last step of the episode/trajectory
            # last gae is basically the accumulated TD from the end to current step
            # ie: exponentially-discounted cumulative sum of 1-step TD errors (delta) 
            # from the end of the trajectory backward to the current step t.

            # we avoid monte carlo's high variance 
            # by avoiding learning wrong conclusion from future error with lambda discount 
            # for future rewards.
            A_t = delta + gamma * gae_lambda * last_gae * next_non_terminal

            self.advantages[t] = A_t
            
            # update last_gae
            last_gae = A_t

        # compute returns for every step, where G_t = A_t + V(s)
        # since Q_hat = G_t
        # use element wise addtion
        self.returns = self.advantages + self.values


    def get_batches(self, batch_size: int = 64) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield randomized PyTorch Tensor mini-batches for SGD optimization epochs."""
        # randomize indices
        # set of observation with associated advantage, return, 
        # log_prob will be put will be put in a batch out of 64 batch

        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices) # shuffle step indices

        for start in range(0, self.buffer_size, batch_size):
            batch_idx = indices[start: start+batch_size] # slice of random indices

            yield {
                "obs": torch.as_tensor(self.observations[batch_idx], device=self.device),
                "actions": torch.as_tensor(self.actions[batch_idx], device=self.device),
                "old_log_probs": torch.as_tensor(self.log_probs[batch_idx], device=self.device), # PPO
                "advantages": torch.as_tensor(self.advantages[batch_idx], device=self.device), # for PPO loss function
                "returns": torch.as_tensor(self.returns[batch_idx], device=self.device), # for critic network MSE loss function
                "values": torch.as_tensor(self.values[batch_idx], device=self.device) # for critic
            }
