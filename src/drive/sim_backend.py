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

def clamp(value, low, high):
    return max(low, min(high, value))

class SimulatedCar(CarBackend):
    def __init__(self, wheelbase: float = 0.25, max_speed: float = 5.0, drag: float = 0.1, max_steering_angle: float = math.pi/3) -> None:
        self.wheelbase: float = wheelbase
        self.max_speed: float = max_speed
        self.max_steering_angle = max_steering_angle
        self.drag: float = drag
        
        self.t: float = 0.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.heading: float = 0.0       # Radians
        self.speed: float = 0.0         # m/s
        self.steering: float = 0.0      # Radians
        
        self.active_command: CarCommand = CarCommand()
        self.battery: float = 100.0

        self.mode: DriveMode = DriveMode.MANUAL
        self.is_connected: bool = False

    def set_mode(self, mode: DriveMode) -> None:
        """Set the drive mode of the simulated car."""
        self.mode = mode

    def connect(self) -> None:
        """Initialize simulated parameters."""
        self.is_connected = True

    def send_command(self, command: CarCommand) -> None:
        """Send target actuator outputs to the simulation state tracker."""
        self.active_command = command

    def update(self, dt: float) -> None:
        """Update simulated physics state based on kinematic bicycle model equations."""
        # 1. Read self . active_command . throttle , steering , brake
        cmd = self.active_command
        throttle, steering, brake = cmd.throttle, cmd.steering, cmd.brake

        # Compute acceleration a
        A_scale = 3.0
        D_scale = 8.0
        accel = (throttle * A_scale) - (brake * D_scale) - (self.drag * self.speed * abs(self.speed))

        # Integrate velocity self .speed , clamping to self .max_speed
        self.speed = clamp(self.speed + accel * dt, -self.max_speed, self.max_speed)

        # Integrate coordinates self .x , self .y
        # Use back wheel as the reference point.
        self.x = self.x + (self.speed * math.cos(self.heading)) * dt
        self.y = self.y + (self.speed * math.sin(self.heading)) * dt
        
        # Integrate heading self . heading
        steering_angle = clamp(steering*self.max_steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.steering = steering_angle
        self.heading = (self.heading + (self.speed/self.wheelbase) * math.tan(steering_angle) * dt) % (2*math.pi)

        # Reduce self . battery
        battery_decay_rate = 0.1
        base_consumption = 0.05
        self.battery = max(0.0, self.battery - (base_consumption + battery_decay_rate*abs(self.speed))*dt)

        self.t = self.t + dt


    def telemetry(self) -> CarTelemetry:
        """Construct and return the current simulated telemetry snapshot."""
        return CarTelemetry(
            time_s=self.t,
            mode=self.mode,
            speed_mps=self.speed,
            heading_deg=math.degrees(self.heading),
            x=self.x,
            y=self.y,
            steering_angle_deg=math.degrees(self.steering),
            battery_pct=self.battery,
            lidar_scan=None,
            crashed=False
        )
