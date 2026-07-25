# """Obstacle layout database: tracking simulated hurdles in the virtual world.
#
# This class acts as a localized database for static or dynamic obstacles (such as traffic cones
# and barricades). It allows the virtual Lidar simulator (lidar_sim) to run collision queries
# and proximity checks to calculate synthetic distance sensors measurements.
# """

from typing import List, Tuple

class ObstacleMap:
    def __init__(self) -> None:
        self.obstacles: List[Tuple[float, float, float]] = [] # x, y, radius for circles

    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        """Add a static circular obstacle to the simulated environment."""
        pass

    def get_obstacles_in_range(self, x: float, y: float, range_m: float) -> List[Tuple[float, float, float]]:
        """Return a list of obstacles within a radius from coordinate (x, y)."""
        pass

    def is_obstacle(self, x: float, y: float) -> bool:
        """Return if the point is part of an obstacle"""
        return False