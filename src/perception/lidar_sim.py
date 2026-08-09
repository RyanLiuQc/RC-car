# """The virtual Lidar sensor simulator: querying world geometry for simulated ranges.
#
# This simulator acts as a mock distance sensor for offline testing. It implements
# the LidarDevice contract, querying the active backend telemetry coordinates and
# the environment map (ObstacleMap) to calculate raycast intersection distances.
# """

import math

from typing import List
from src.common.sensor import LidarDevice
from src.common.backend import CarBackend
from src.common.types import LidarScan
from src.environment.obstacles import ObstacleMap
from src.environment.track import Track

class LidarSimulator(LidarDevice):
    def __init__(self, obstacle_map: ObstacleMap, backend: CarBackend, track: Track = None, num_rays: int = 3, max_range_m: float = 5.0) -> None:
        self.obstacle_map: ObstacleMap = obstacle_map
        self.backend: CarBackend = backend
        self.track: Track = track
        self.num_rays: int = num_rays
        self.max_range_m: float = max_range_m # max range of the rays in meters

        if self.num_rays == 3:
            self.angles_deg = [-30.0, 0.0, 30.0]
        else:
            # Generates evenly spaced angles across a 60-degree front arc
            fov_deg = 60.0
            step = fov_deg / (self.num_rays - 1)
            # angle of every ray
            self.angles_deg = [-fov_deg / 2.0 + i * step for i in range(self.num_rays)] 


    def read_scan(self) -> LidarScan:
        """
        Query vehicle telemetry coordinates from the backend and compute raycast distances.
        Returns a LidarScan object.
        """
        # Fetch current coordinates from the backend contract
        telemetry = self.backend.telemetry()
        car_x = telemetry.x
        car_y = telemetry.y
        car_heading = telemetry.heading_deg

        ranges_m = []

        # Step size for raymarching in meters (5cm resolution)
        step_m = 0.05
        
        # Compute real intersection distances using car_x, car_y, car_heading with raymarching
        for ray_angle in self.angles_deg:
            abs_ray_angle = math.radians(ray_angle + car_heading)

            # unit vector for the direction of the ray
            unit_directional_vector = (math.cos(abs_ray_angle), math.sin(abs_ray_angle))

            hit_dist = self.max_range_m # hit distance (default to max range first)
            num_steps = int(self.max_range_m / step_m)
            for i in range(num_steps):
                d = step_m * i
                p = (car_x + d*unit_directional_vector[0], car_y + d*unit_directional_vector[1])

                # Check 1: Did the ray hit a boundary?
                if self.track and not self.track.is_within_boundaries(p[0], p[1]):
                    hit_dist = d
                    break

                # Check 2: Did the ray hit an obstacle?
                if self.obstacle_map and self.obstacle_map.is_obstacle(p[0], p[1]):
                    hit_dist = d
                    break

            ranges_m.append(hit_dist)
            
        return LidarScan(
            time_s=telemetry.time_s,
            angles_deg=self.angles_deg,
            ranges_m=ranges_m
        )
