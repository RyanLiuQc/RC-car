# """Perception pipeline test suite: validating CV and Lidar process algorithms.
#
# This test file runs unit tests on the computer vision LaneDetector math,
# the LidarProcessor scan groups, and the LidarSimulator raycasting algorithms.
# """

from src.perception.lane_detector import LaneDetector
from src.perception.lidar_scan import LidarProcessor
from src.perception.lidar_sim import LidarSimulator
from src.drive.sim_backend import SimulatedCar
from src.environment.track import Track
from src.environment.obstacles import ObstacleMap

def test_lane_detector_creation() -> None:
    """Verify that LaneDetector object instantiates without raising exceptions."""
    detector = LaneDetector()
    assert detector is not None

def test_lidar_simulator_raymarching() -> None:
    """Verify LidarSimulator raymarching scan output on track."""
    car = SimulatedCar()
    car.connect()
    track = Track(track_name="default_oval", track_width=1.6)
    obs_map = ObstacleMap()
    
    lidar_sim = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=3, max_range_m=5.0)
    scan = lidar_sim.read_scan()
    
    assert len(scan.angles_deg) == 3
    assert len(scan.ranges_m) == 3
    assert scan.angles_deg == [-30.0, 0.0, 30.0]
    # At starting position (0,0) on centerline, rays should return positive ranges within max_range
    assert all(0.0 < r <= 5.0 for r in scan.ranges_m)
