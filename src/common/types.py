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
    """Actuation target command containing normalized throttle, steering, and braking requests."""
    throttle: float = 0.0  # Range: [-1.0 to 1.0]. Ref: 0.0 is idle, 1.0 is full forward.
    steering: float = 0.0  # Range: [-1.0 to 1.0]. Ref: 0.0 is straight, -1.0 is full left, 1.0 is full right.
    brake: float = 0.0     # Range: [0.0 to 1.0]. Ref: 0.0 is released, 1.0 is full electronic brake force.

@dataclass
class LidarScan:
    """Obstacle scanning points consisting of measured ranges at designated sweep angles."""
    time_s: float = 0.0
    angles_deg: List[float] = field(default_factory=list)  # Scan angles relative to heading
    ranges_m: List[float] = field(default_factory=list)    # Measured obstacle distances

@dataclass
class Waypoint:
    """A target navigation coordinate waypoint with target speed properties."""
    x: float                  # Global coordinates in meters (m). Ref: relative to track origin (0.0, 0.0).
    y: float                  # Global coordinates in meters (m). Ref: relative to track origin (0.0, 0.0).
    target_speed_mps: float = 1.5 # Target cruise speed in meters per second (m/s).

@dataclass
class Path:
    """A collection of sequential waypoints forming a route or looping track."""
    waypoints: List[Waypoint] = field(default_factory=list)
    closed_loop: bool = True  # True if path forms a loop track

@dataclass
class LaneLineState:
    """Chassis position estimates relative to visual road lane markers."""
    detected: bool = False
    center_offset_m: float = 0.0
    lane_heading_deg: float = 0.0

@dataclass
class FrenetState:
    """Vehicle coordinates mapped relative to the track reference centerline."""
    s: float                  # Longitudinal distance traveled along centerline in meters (m). Ref: starts at 0.0 at start line.
    d: float                  # Lateral cross-track error offset from centerline in meters (m). Ref: 0.0 is center, positive is left.
    heading_error_deg: float  # Yaw orientation error relative to the track tangent line in degrees. Range: [-180.0, 180.0].

@dataclass
class CarTelemetry:
    """Comprehensive snapshot of the vehicle state and diagnostic values."""
    time_s: float             # Elapsed system time in seconds (s). Ref: starts at 0.0 on boot.
    mode: DriveMode           # Current active operational drive state enum.
    speed_mps: float          # Linear forward velocity in meters per second (m/s). Ref: positive is forward, negative is reverse.
    heading_deg: float        # Absolute yaw orientation angle in degrees. Ref: [0.0, 360.0) relative to positive X-axis (east).
    x: float                  # Current global position coordinates X in meters (m). Ref: relative to track origin (0.0, 0.0).
    y: float                  # Current global position coordinates Y in meters (m). Ref: relative to track origin (0.0, 0.0).
    steering_angle_deg: float # Actual physical front wheel steer in degrees. Ref: 0.0 is straight, negative is left, positive is right.
    battery_pct: float        # Remaining onboard battery capacity percentage. Range: [0.0, 100.0].
    lidar_scan: LidarScan     # Structured obstacle Lidar scan readings.
    crashed: bool             # Crash status flag. Ref: True if vehicle boundaries intersect track walls or obstacles.
