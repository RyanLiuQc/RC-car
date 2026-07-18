# """Track geometry and boundaries: defining the simulated world layout.
#
# This file stores references to track lanes, boundary walls, and navigation checkpoints.
# It exposes spatial helpers to locate check-points and query coordinate intersections,
# allowing the physical simulated vehicle and sensor suites to verify their positions on track.
# """

from typing import List, Tuple
from src.common.types import FrenetState

class Track:
    def __init__(self, track_name: str = "default_oval") -> None:
        self.track_name: str = track_name
        self.waypoints: List[Tuple[float, float]] = []
        self.boundaries: List[Tuple[float, float, float, float]] = [] # Lines representing walls/lines

    def get_nearest_waypoint(self, x: float, y: float) -> Tuple[float, float]:
        """Find the coordinates of the closest checkpoint waypoint on the track."""
        pass

    def is_within_boundaries(self, x: float, y: float) -> bool:
        """Check if a coordinate position is within the track boundaries."""
        pass

    def cartesian_to_frenet(self, x: float, y: float, heading_deg: float) -> FrenetState:
        """
        Convert Cartesian global coordinates (x, y, heading) to track-relative Frenet (s, d, heading_error).
        """
        pass
