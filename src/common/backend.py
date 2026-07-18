# """The hardware/simulation abstraction: defining the contract for car operations.
#
# This interface dictates the primary commands (connect, set_controls, set_brake, update,
# and telemetry) that any backend (whether a physical servo board, a kinematic physics model,
# or a remote communication link) must implement. This establishes a strict API boundary,
# allowing the high-level controllers to steer the car without knowing what physically drives it.
# """

from abc import ABC, abstractmethod
from src.common.types import CarTelemetry

class CarBackend(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Initialize connection to hardware or simulation."""
        pass

    @abstractmethod
    def set_controls(self, throttle: float, steering: float) -> None:
        """
        Send inputs to the vehicle motors/servos.
        - throttle: -1.0 (full reverse) to 1.0 (full forward)
        - steering: -1.0 (full left lock) to 1.0 (full right lock)
        """
        pass

    @abstractmethod
    def set_brake(self, force: float) -> None:
        """Apply electronic braking force (0.0 to 1.0)."""
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Pump communications / step physics."""
        pass

    @abstractmethod
    def telemetry(self) -> CarTelemetry:
        """Return the current telemetry snapshot."""
        pass
