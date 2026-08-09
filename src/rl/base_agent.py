# """Abstract RL Agent Interface: defining the common contract for reinforcement learning policies.
#
# This file declares the BaseAgent abstract base class contract (select_action, train_step,
# save, load). All concrete agent algorithms (RandomAgent, PPOAgent, SACAgent) inherit from this
# contract, enabling plug-and-play policy switching in training and inference pipelines.
# """

from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return raw action array for env.step()."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, trajectory_buffer: dict) -> dict:
        """Perform policy update step using collected experience."""
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model weights to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model weights from disk."""
        raise NotImplementedError
