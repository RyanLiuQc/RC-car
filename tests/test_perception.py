# """
# Purpose: Unit tests for CV lane detection algorithms and lidar scanning functions.
# """

from src.perception.lane_detector import LaneDetector
from src.perception.lidar_scan import LidarProcessor

def test_lane_detector_creation() -> None:
    """Verify that LaneDetector object instantiates without raising exceptions."""
    detector = LaneDetector()
    assert detector is not None
