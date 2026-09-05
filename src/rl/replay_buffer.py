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
        self.ptr: int = 0 # next idx available
        self.size: int = 0
        
        # Initialize numpy array buffers for obs, actions, rewards, next_obs, and terminated flags.
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        """Store a single step transition tuple into the circular replay buffer."""

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity) 

        # if self.size = min(self.ptr + 1, self.capacity) 
        # Problem at size = capacity - 1, next add will transition ptr = 0 (modulo) -> self.size = 0
        # and if sample() is called, indices will be chosen from a=self.size=0 -> array = [] empty list to choose from


    def sample(self, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Sample a random mini-batch of transitions for gradient updates."""
        indices = np.random.choice(a=self.size, size=batch_size)

        # return dict with the randomized 
        return {
            "observations": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "terminated": self.terminated[indices]
        }