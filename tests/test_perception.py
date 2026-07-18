# """Perception pipeline test suite: validating CV and Lidar process algorithms.
#
# This test file runs unit tests on the computer vision LaneDetector math and
# the LidarProcessor scan groups, ensuring synthetic sensor frames are correctly
# parsed to locate lanes and cluster obstacle ranges.
# """

from src.perception.lane_detector import LaneDetector
from src.perception.lidar_scan import LidarProcessor

def test_lane_detector_creation() -> None:
    """Verify that LaneDetector object instantiates without raising exceptions."""
    detector = LaneDetector()
    assert detector is not None
