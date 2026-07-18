# """
# Purpose: Define obstacle layout maps and collision detection checks.
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
