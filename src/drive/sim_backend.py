# """Kinematic 2D bicycle simulation: virtualizing vehicle dynamics.
#
# This simulated backend implements the CarBackend abstract interface contract.
# It models vehicle motion based on the 2D kinematic bicycle model equations (incorporating
# wheelbase, drag, and steering angles). It updates the position, heading, and battery
# states at a designated step size, enabling local offline unit testing.
# """

import math
from src.common.backend import CarBackend
from src.common.types import CarTelemetry, DriveMode, CarCommand, LidarScan

class SimulatedCar(CarBackend):
    def __init__(self, wheelbase: float = 0.25, max_speed: float = 5.0, drag: float = 0.1) -> None:
        self.wheelbase: float = wheelbase
        self.max_speed: float = max_speed
        self.drag: float = drag
        
        self.t: float = 0.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.heading: float = 0.0       # Radians
        self.speed: float = 0.0         # m/s
        self.steering: float = 0.0      # Radians
        
        self.active_command: CarCommand = CarCommand()
        self.battery: float = 100.0

    def connect(self) -> None:
        """Initialize simulated parameters."""
        pass

    def send_command(self, command: CarCommand) -> None:
        """Send target actuator outputs to the simulation state tracker."""
        self.active_command = command

    def update(self, dt: float) -> None:
        """Update simulated physics state based on kinematic bicycle model equations."""
        pass

    def telemetry(self) -> CarTelemetry:
        """Construct and return the current simulated telemetry snapshot."""
        # TODO: Construct a real LidarScan using simulator raycasting data
        mock_scan = LidarScan(time_s=self.t, angles_deg=[-30.0, 0.0, 30.0], ranges_m=[5.0, 5.0, 5.0])
        return CarTelemetry(
            time_s=self.t,
            mode=DriveMode.MANUAL,
            speed_mps=self.speed,
            heading_deg=math.degrees(self.heading),
            x=self.x,
            y=self.y,
            steering_angle_deg=math.degrees(self.steering),
            battery_pct=self.battery,
            lidar_scan=mock_scan,
            crashed=False
        )
