# """
# Purpose: Unit tests to verify the simulation backend, controller commands, and safety locks.
# """

from src.drive.sim_backend import SimulatedCar
from src.drive.controller import CarController

def test_simulated_car_creation() -> None:
    """Verify that SimulatedCar instantiation sets correct starting conditions."""
    car = SimulatedCar()
    assert car.speed == 0.0
    assert car.battery == 100.0
