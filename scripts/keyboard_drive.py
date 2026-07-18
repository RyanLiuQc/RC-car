# """Keyboard teleoperation drive cockpit: driving the car with manual terminal keystrokes.
#
# This script launches an interactive terminal loop. It intercepts keystroke inputs,
# parses them using the KeyboardTeleop command mapper, and forwards steering and speed controls
# directly to the vehicle backend. It acts as the primary manual testing environment.
# """

import sys
from src.drive.sim_backend import SimulatedCar
from src.drive.controller import CarController
from src.tools.keyboard_teleop import KeyboardTeleop

def main() -> None:
    """Run keyboard teleoperation event loops in simulated or physical environments."""
    pass

if __name__ == "__main__":
    main()
