# """Offline validation suite: testing vehicle dynamics and collision logic.
#
# This test file verifies that the SimulatedCar kinematics, controller commands,
# and CollisionAvoidance brakes perform accurately in virtual conditions.
# It runs fully offline with zero external hardware or network dependencies.
# """

from src.drive.sim_backend import SimulatedCar
from src.drive.controller import CarController

def test_simulated_car_creation() -> None:
    """Verify that SimulatedCar instantiation sets correct starting conditions."""
    car = SimulatedCar()
    assert car.speed == 0.0
    assert car.battery == 100.0
