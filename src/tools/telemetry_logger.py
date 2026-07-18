# """
# Purpose: Log telemetry frames to a CSV file for offline graphing and diagnostics.
# """

from src.common.types import CarTelemetry

class TelemetryLogger:
    def __init__(self, output_path: str = "telemetry.csv") -> None:
        self.output_path = output_path
        self.header_written = False

    def log(self, telemetry: CarTelemetry) -> None:
        """Append a telemetry record frame into the output CSV file."""
        pass
