# """Keyboard command mapper: translating keys to motor commands.
#
# This teleoperation tool parses user input characters (like WASD or arrow keys) in the terminal
# to output normalized vehicle speed and direction controls. It serves as the primary cockpit driver,
# enabling developers to manual-override or manually drive the simulated/physical car.
# """

from typing import Tuple

class KeyboardTeleop:
    def __init__(self, throttle_step: float = 0.1, steering_step: float = 0.1) -> None:
        self.throttle_step: float = throttle_step
        self.steering_step: float = steering_step
        self.throttle: float = 0.0
        self.steering: float = 0.0

    def parse_key(self, key_char: str) -> Tuple[float, float]:
        """
        Evaluate key input characters (e.g. WASD, arrow keys) to compute target controls.
        Returns a tuple of (throttle, steering) ranges [-1.0 .. 1.0].
        """
        pass

    def reset(self) -> None:
        """Reset internal speed and steering controls back to neutral/0.0."""
        pass
