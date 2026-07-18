# """
# Purpose: Map terminal keystrokes to normalized vehicle steering and throttle controls.
# """

from typing import Tuple

class KeyboardTeleop:
    def __init__(self, throttle_step: float = 0.1, steering_step: float = 0.1) -> None:
        self.throttle_step = throttle_step
        self.steering_step = steering_step
        self.throttle = 0.0
        self.steering = 0.0

    def parse_key(self, key_char: str) -> Tuple[float, float]:
        """
        Evaluate key input characters (e.g. WASD, arrow keys) to compute target controls.
        Returns a tuple of (throttle, steering) ranges [-1.0 .. 1.0].
        """
        pass

    def reset(self) -> None:
        """Reset internal speed and steering controls back to neutral/0.0."""
        pass
