# """Shared telemetry dataclasses and enums: the system's common vocabulary.
#
# This file defines the telemetry structures (speed, heading, battery, sonar ranges, and positions)
# and mode states (MANUAL, AUTONOMOUS, ESTOP) passed across all modules. It has no dependencies,
# serving as the lightweight dictionary that perception, drive, state, and loggers use to communicate.
# """

from dataclasses import dataclass, field
from enum import Enum
from typing import List

class DriveMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"
    ESTOP = "ESTOP"

@dataclass
class CarCommand:
    throttle: float = 0.0  # Range: -1.0 (full reverse) to 1.0 (full forward)
    steering: float = 0.0  # Range: -1.0 (full left lock) to 1.0 (full right lock)
    brake: float = 0.0     # Range: 0.0 (released) to 1.0 (full braking force)

@dataclass
class LidarScan:
    time_s: float = 0.0
    angles_deg: List[float] = field(default_factory=list)  # Scan angles relative to heading
    ranges_m: List[float] = field(default_factory=list)    # Measured obstacle distances

@dataclass
class Waypoint:
    x: float
    y: float
    target_speed_mps: float = 1.5

@dataclass
class Path:
    waypoints: List[Waypoint] = field(default_factory=list)
    closed_loop: bool = True  # True if path forms a loop track

@dataclass
class LaneLineState:
    detected: bool = False
    center_offset_m: float = 0.0
    lane_heading_deg: float = 0.0

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
    lidar_scan: LidarScan     # Structured obstacle Lidar scan readings
    crashed: bool
