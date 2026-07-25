"""Unit tests for SimulatedCar 2D bicycle kinematics, controls, and telemetry."""

import math
import pytest
from src.drive.sim_backend import SimulatedCar
from src.common.types import CarCommand, DriveMode


@pytest.fixture
def car() -> SimulatedCar:
    """Fixture providing a connected SimulatedCar instance."""
    sim = SimulatedCar(wheelbase=0.25, max_speed=5.0, drag=0.1)
    sim.connect()
    return sim


def test_simulated_car_initial_state(car: SimulatedCar) -> None:
    """Verify default initial values for SimulatedCar."""
    assert car.is_connected is True
    assert car.speed == 0.0
    assert car.heading == 0.0
    assert car.x == 0.0
    assert car.y == 0.0
    assert car.battery == 100.0


def test_simulated_car_acceleration(car: SimulatedCar) -> None:
    """Verify velocity updates under throttle input."""
    # Apply full forward throttle (1.0)
    car.send_command(CarCommand(throttle=1.0, steering=0.0, brake=0.0))
    car.update(dt=0.1)

    # Speed should increase above 0
    assert car.speed > 0.0
    # Position X should move forward (heading is 0)
    assert car.x > 0.0
    assert car.y == pytest.approx(0.0, abs=1e-5)


def test_simulated_car_braking(car: SimulatedCar) -> None:
    """Verify braking decelerates the vehicle speed."""
    # Accelerate first
    car.send_command(CarCommand(throttle=1.0, steering=0.0, brake=0.0))
    car.update(dt=0.5)
    speed_before_brake = car.speed

    # Apply full brake
    car.send_command(CarCommand(throttle=0.0, steering=0.0, brake=1.0))
    car.update(dt=0.1)

    assert car.speed < speed_before_brake


def test_simulated_car_steering_kinematics(car: SimulatedCar) -> None:
    """Verify heading angle updates when steering is applied with positive speed."""
    # Give car initial speed and apply left steer (+1.0)
    car.speed = 2.0
    car.send_command(CarCommand(throttle=0.5, steering=1.0, brake=0.0))
    car.update(dt=0.1)

    # Heading should change from 0.0
    assert car.heading != 0.0


def test_simulated_car_heading_modulo(car: SimulatedCar) -> None:
    """Verify heading remains normalized within [0, 2*pi)."""
    car.heading = 2 * math.pi + 0.5
    # Force update tick to trigger heading normalization if implemented
    car.speed = 1.0
    car.update(dt=0.01)

    assert 0.0 <= car.heading < 2 * math.pi


def test_simulated_car_telemetry_snapshot(car: SimulatedCar) -> None:
    """Verify telemetry snapshot constructs valid properties."""
    telemetry = car.telemetry()
    assert telemetry.speed_mps == car.speed
    assert telemetry.mode == DriveMode.MANUAL
    assert telemetry.battery_pct == car.battery
