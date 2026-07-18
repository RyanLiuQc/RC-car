# """
# Purpose: Define the virtual driving track boundaries, lane curves, and reference waypoints.
# """

from typing import List, Tuple

class Track:
    def __init__(self, track_name: str = "default_oval"):
        self.track_name = track_name
        self.waypoints: List[Tuple[float, float]] = []
        self.boundaries: List[Tuple[float, float, float, float]] = [] # Lines representing walls/lines

    def get_nearest_waypoint(self, x: float, y: float) -> Tuple[float, float]:
        """Find the coordinates of the closest checkpoint waypoint on the track."""
        pass

    def is_within_boundaries(self, x: float, y: float) -> bool:
        """Check if a coordinate position is within the track boundaries."""
        pass
