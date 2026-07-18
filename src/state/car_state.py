# """
# Purpose: Track statistics, run estimates, and manage health states using telemetry frames.
# """

from src.common.types import CarTelemetry

class CarStateManager:
    def __init__(self) -> None:
        self.total_distance_m: float = 0.0
        self.last_telemetry: CarTelemetry = None
        self.speeds: list[float] = []

    def update(self, telemetry: CarTelemetry) -> None:
        """Process a new telemetry snapshot to compute mileage, average speed, and monitor battery."""
        pass

    def get_average_speed(self) -> float:
        """Calculate the running average speed of the vehicle."""
        pass

    def is_battery_low(self) -> bool:
        """Check if battery percentage is below critical warning thresholds."""
        pass
