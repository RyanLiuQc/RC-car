# """Track geometry and boundaries: defining the simulated world layout.
#
# This file stores references to track lanes, boundary walls, and navigation checkpoints.
# It exposes spatial helpers to locate check-points and query coordinate intersections,
# allowing the physical simulated vehicle and sensor suites to verify their positions on track.
# """

import math
from typing import List, Tuple
from src.common.types import FrenetState

class Track:
    def __init__(self, track_name: str = "default_oval", track_width: float = 1.6) -> None:
        self.track_name: str = track_name
        # waypoint can eventually store a width value as the 3rd float for more realistic real world sim
        self.waypoints: List[Tuple[float, float]] = [
            (0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (5.0, 1.0),
            (5.0, 3.0), (4.0, 4.0), (2.0, 4.0), (0.0, 4.0),
            (-1.0, 3.0), (-1.0, 1.0)
        ] # Centerline coordinate path (x, y) for trajectory tracking and Frenet projection

        # Building Wall/curb line segments (x1, y1, x2, y2) for collision checks and Lidar raycasting
        self.track_width = track_width
        left_boundary, right_boundary = generate_track_boundaries(
            waypoints=self.waypoints, track_width=track_width)
        left_segments = [
            (p1[0],p1[1],p2[0],p2[1]) for p1,p2 in zip(left_boundary, left_boundary[1:]+[left_boundary[0]])
        ]
        right_segments = [
            (p1[0],p1[1],p2[0],p2[1]) for p1,p2 in zip(right_boundary, right_boundary[1:]+[right_boundary[0]])
        ]
        
        self.boundaries: List[Tuple[float, float, float, float]] = left_segments + right_segments 

    def get_nearest_waypoint_index(self, x: float, y: float) -> int:
        """Find the index of the closest checkpoint waypoint on the track."""
        min_dist_square = float("inf")
        nearest_idx = 0
        for i, (w_x, w_y) in enumerate(self.waypoints):
            dist_sq = (w_x - x)**2 + (w_y - y)**2
            if dist_sq < min_dist_square: 
                min_dist_square = dist_sq 
                nearest_idx = i
        return nearest_idx

    def get_nearest_waypoint(self, x: float, y: float) -> Tuple[float, float]:
        """Find the coordinates of the closest checkpoint waypoint on the track."""
        idx = self.get_nearest_waypoint_index(x, y)
        return self.waypoints[idx]

    def _get_distance_to_segment(self, 
                                 x: float, 
                                 y: float, 
                                 A: Tuple[float, float], # endpoint
                                 B: Tuple[float, float] # endpoint
                                 ) -> float:
        """Compute point to segment distance with projection"""
        v = (x-A[0],y-A[1]) # segment from waypoint to car
        u = (B[0]-A[0], B[1]-A[1])

        dot_product = v[0]*u[0] + v[1]*u[1]
        u_length_squared = u[0]**2 + u[1]**2

        # compute scaling factor of the projection vector
        scale = dot_product / u_length_squared if u_length_squared != 0 else float('inf')
        t = max(0.0, min(1.0, dot_product / u_length_squared))

        # find point on segment closest to car. (creates perpendicular line)
        P = (A[0] + t*u[0], A[1] + t**u[1])

        dist = math.sqrt((A[0]-P[0])**2 + (A[1]-P[1])**2)

        return dist
        
    
    def _get_closest_segment_to_point_idx(self, nearest_waypoint_idx):
        """Find closest segment to point idx using projection"""
        k = nearest_waypoint_idx

        n = len(self.waypoints)
        prev_idx = (k-1) % n
        next_idx = (k+1) % n

        for i in [prev_idx, next_idx]:
            pass


        

        
        

    def is_within_boundaries(self, x: float, y: float) -> bool:
        """Check if a coordinate position is within the track boundaries."""
        # TODO: does not work. nearest_waypoint waypoint is unreliable for comparing distance
        # get nearest waypoint index
        waypoint_idx = self.get_nearest_waypoint_index(x,y)

       # get closest segment index to the car
       closest_segment_idx = self._get_closest_segment_to_point_idx()

       # compute distance d from given indexes
        dist = 0

        # check if distance <= track_width
        return dist <= self.track_width/2

    def cartesian_to_frenet(self, x: float, y: float, heading_deg: float) -> FrenetState:
        """
        Convert Cartesian global coordinates (x, y, heading) to track-relative Frenet (s, d, heading_error).
        """
        # 1. Find closest waypoint index i
        w_x, w_y = self.get_nearest_waypoint(x,y)
        # 2. Extract segment vector v = waypoints[i+1] - waypoints[i]
        

        # 3. Extract car vector u = (x,y) - waypoints[i]
        # 4. Project u onto v to find projection factor t
        # 5. Compute lateral distance d = distance to projected point
        # 6. Set sign of d using cross product : v_x * u_y - v_y * u_x
        # 7. Compute cumulative distance s along path
        # 8. Compute heading error delta_theta = heading_deg - segment_tangent_angle
        # 9. Return FrenetState ( s =s , d =d , heading_error_deg = delta_theta )
        pass


def generate_track_boundaries(
    waypoints: List[Tuple[float, float]], track_width: float = 1.6
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Generate left (outer) and right (inner) boundary coordinates 
    based on a list of centerline waypoints.
    """
    left_boundary: List[Tuple[float, float]] = []
    right_boundary: List[Tuple[float, float]] = []
    n: int = len(waypoints)
    
    for i in range(n):
        x, y = waypoints[i]
        
        # Get adjacent waypoints (handles closed-loop tracks)
        prev_x, prev_y = waypoints[(i - 1) % n]
        next_x, next_y = waypoints[(i + 1) % n]
        
        # Calculate tangent vector at this point
        dx: float = next_x - prev_x
        dy: float = next_y - prev_y
        length: float = math.sqrt(dx**2 + dy**2)
        
        # Unit normal vector pointing to the left
        nx: float = -dy / length
        ny: float = dx / length
        
        # Offset left and right by half track width (0.8m)
        half_w: float = track_width / 2.0
        left_boundary.append((x + nx * half_w, y + ny * half_w))
        right_boundary.append((x - nx * half_w, y - ny * half_w))
        
    return left_boundary, right_boundary
