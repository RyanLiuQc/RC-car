# """Experience Replay Buffer: storing transition tuples for off-policy (SAC) RL algorithms.
#
# This data structure stores transition tuples (obs, action, reward, next_obs, terminated)
# in a fixed-size circular ring buffer. It allows off-policy algorithms (like Soft Actor-Critic - SAC)
# to sample random mini-batches for sample-efficient gradient updates.
# """

import numpy as np
from typing import Tuple, Dict, Any

class ReplayBuffer:
    def __init__(self, capacity: int = 100000, obs_dim: int = 6, action_dim: int = 2) -> None:
        self.capacity: int = capacity
        self.ptr: int = 0
        self.size: int = 0
        
        # TODO: Initialize numpy array buffers for obs, actions, rewards, next_obs, and terminated flags.

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        """Store a single step transition tuple into the circular replay buffer."""
        pass

    def sample(self, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Sample a random mini-batch of transitions for gradient updates."""
        pass