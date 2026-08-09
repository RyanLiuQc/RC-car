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
        if track_name == "s_curve":
            self.waypoints = [
                (0.0, 0.0), (3.0, 0.0), (5.0, 1.0), (5.0, 3.0),
                (3.0, 4.0), (1.0, 5.0), (1.0, 7.0), (3.0, 8.0),
                (6.0, 8.0), (8.0, 6.0), (8.0, 2.0), (6.0, -1.0),
                (3.0, -1.0)
            ]
        elif track_name == "figure_eight":
            self.waypoints = [
                (0.0, 0.0), (2.0, 1.0), (3.0, 3.0), (2.0, 5.0),
                (0.0, 6.0), (-2.0, 5.0), (-3.0, 3.0), (-2.0, 1.0),
                (0.0, 0.0), (2.0, -1.0), (3.0, -3.0), (2.0, -5.0),
                (0.0, -6.0), (-2.0, -5.0), (-3.0, -3.0), (-2.0, -1.0)
            ]
        else: # "default_oval"
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
                                 ) -> Tuple[float, Tuple[float,float], float]:
        """Compute point to segment distance with projection
        return: distance, nearest_point_on_segment (e.i. projected point), t_factor
        """
        u = (x-A[0],y-A[1])         # tangent vector of the waypoint
        v = (B[0]-A[0], B[1]-A[1])  # segment from waypoint to car

        dot_product = u[0]*v[0] + u[1]*v[1]
        v_length_squared = v[0]**2 + v[1]**2 # the norm squared

        # compute scaling factor of the projection vector, ensure no division by zero
        if v_length_squared == 0:
            t = 0
        else:
            # clamping to avoid vector v to scale to infinity, making distance potentially shorter
            t = max(0.0, min(1.0, dot_product / v_length_squared)) 

        # find point on segment closest to car. (creates perpendicular line)
        P = (A[0] + t*v[0], A[1] + t*v[1])

        dist = math.sqrt((x-P[0])**2 + (y-P[1])**2)

        return dist, P, t

    def _get_closest_segment_info(
            self, x: float, y: float
            ) -> Tuple[int, float, Tuple[float,float],float]: 
        """Get index of closest waypoint segment, distance car to segment 
        and scaling factor of the direction vector on the segment such that the point on the line
        is the closest point to the car

        Input: x,y
        
        Output: index of segmentt, distance, projection_point, scaling factor
        
        """
        # get nearest waypoint index
        k = self.get_nearest_waypoint_index(x,y)

        # get closest segment index to the car
        n = len(self.waypoints)
        prev_idx = (k-1) % n
        next_idx = (k+1) % n

        # compute distance d from given indexes 
        dist_prev_segment, P_prev, t_prev = self._get_distance_to_segment(
            x, y, 
            self.waypoints[prev_idx], self.waypoints[k]
        )
        dist_next_segment, P_next, t_next = self._get_distance_to_segment(
            x, y,
            self.waypoints[k], self.waypoints[next_idx]
        )

        # dist = min(dist_next_segment, dist_prev_segment)
        # if closest waypoint = k
        # index of next segment = k
        # index of prev segment = k-1
        if dist_prev_segment < dist_next_segment:
            dist = dist_prev_segment
            segment_idx = k-1
            P = P_prev
            t = t_prev

        else:
            dist = dist_next_segment
            segment_idx = k
            P = P_next
            t = t_next


        return segment_idx, dist, P, t
    

    def is_within_boundaries(self, x: float, y: float) -> bool:
        """Check if a coordinate position is within the track boundaries."""
        _, dist, _, _ = self._get_closest_segment_info(x, y)

        # check if distance <= track_width
        return dist <= self.track_width/2

    def cartesian_to_frenet(self, x: float, y: float, heading_deg: float) -> FrenetState:
        """
        Convert Cartesian global coordinates (x, y, heading) to track-relative Frenet (s, d, heading_error).
        """
        # 1. Find closest waypoint index i
        # 2. Extract segment vector v = waypoints[i+1] - waypoints[i]
        # 3. Extract car vector u = (x,y) - waypoints[i]
        # 4. Project u onto v to find projection factor t
        # 5. Compute lateral distance d = distance to projected point
        segment_idx, d_magnitude, P, t = self._get_closest_segment_info(x, y)

        # 6. Set sign of d using cross product : v_x * u_y - v_y * u_x
        n = len(self.waypoints)
        # if v cross u > 0 -> dist points to the left -> positive deviation
        v = ( # B[0]-A[0], B[1]-A[1]
            self.waypoints[(segment_idx+1) % n][0] - self.waypoints[segment_idx][0],
            self.waypoints[(segment_idx+1) % n][1] - self.waypoints[segment_idx][1],
        )
        u = ( # x - A, y - A
            x - self.waypoints[segment_idx][0],
            y - self.waypoints[segment_idx][1]
        )
        cross_product = v[0]*u[1] - v[1]*u[0]
        sign = -1 if cross_product < 0 else 1
        d = sign * abs(d_magnitude)
        
        # 7. Compute cumulative distance s along path
        # arc length = s
        s = 0
        n = len(self.waypoints)
        for i in range(segment_idx): # sum up until the closest segment = waypoint[i], waypoint[(i+1)%n]
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i+1) % n]
            s += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
            # s is summed up to the start of the closest segment, we have to add the remaining segment length

        segment_length = math.sqrt(v[0]**2 + v[1]**2)
        s = s + t*segment_length
        

        # 8. Compute heading error delta_theta = heading_deg - segment_tangent_angle
        # if heading_err > 0: heading to the left
        segment_tangent_angle = math.degrees(math.atan2(v[1], v[0])) # param: (y,x)
        # heading_err in degrees between -180 to 180 (try out different cadrant to understand)
        heading_err = ((heading_deg - segment_tangent_angle + 180) % 360) - 180 


        # 9. Return FrenetState
        return FrenetState(s=s, d=d, heading_error_deg=heading_err)


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
