# """
# Purpose: Define shared data types, enums, and telemetry structures for the RC car.
# """

from dataclasses import dataclass
from enum import Enum
from typing import List

class DriveMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"
    ESTOP = "ESTOP"

@dataclass
class CarTelemetry:
    time_s: float
    mode: DriveMode
    speed_mps: float
    heading_deg: float
    x: float                  # Odometry X position (meters from home)
    y: float                  # Odometry Y position (meters from home)
    steering_angle_deg: float
    battery_pct: float
    sonar_ranges_m: List[float] # [Left, Center, Right] ultrasonic readings
    crashed: bool
