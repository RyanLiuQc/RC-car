# """Random Baseline Agent: uniform random continuous action baseline.
#
# This file implements RandomAgent, which inherits from BaseAgent and samples
# random throttle and steering commands from a uniform distribution [-1.0, 1.0].
# It serves as a benchmark control baseline to compare trained RL algorithms against.
# """

import numpy as np
from src.rl.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, action_dim: int = 2) -> None:
        self.action_dim: int = action_dim

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample random continuous action between [-1.0, 1.0]."""
        return np.random.uniform(low=-1.0, high=1.0, size=(self.action_dim,)).astype(np.float32)

    def train_step(self, trajectory_buffer: dict) -> dict:
        """No training step required for random agent."""
        return {}

    def save(self, filepath: str) -> None:
        """No weights to save for random agent."""
        pass

    def load(self, filepath: str) -> None:
        """No weights to load for random agent."""
        pass