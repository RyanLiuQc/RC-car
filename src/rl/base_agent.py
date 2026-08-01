from abc import ABC, abstractmethod
import numpy as np

class BaseAgent:
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
