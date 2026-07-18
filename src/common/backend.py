# """The hardware/simulation abstraction: defining the contract for car operations.
#
# This interface dictates the primary commands (connect, send_command, update,
# and telemetry) that any backend (whether a physical servo board, a kinematic physics model,
# or a remote communication link) must implement. This establishes a strict API boundary,
# allowing the high-level controllers to steer the car without knowing what physically drives it.
# """

from abc import ABC, abstractmethod
from src.common.types import CarTelemetry, CarCommand

class CarBackend(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Initialize connection to hardware or simulation."""
        pass

    @abstractmethod
    def send_command(self, command: CarCommand) -> None:
        """
        Send target actuator outputs (throttle, steering, brake) to the vehicle.
        - command: CarCommand containing throttle, steering, and brake targets.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Pump communications / step physics."""
        pass

    @abstractmethod
    def telemetry(self) -> CarTelemetry:
        """Return the current telemetry snapshot."""
        pass
