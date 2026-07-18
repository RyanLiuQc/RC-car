# """The Lidar sensor device abstraction: defining the contract for range sensors.
#
# This interface dictates the read_scan method that any range-finding sensor device
# (whether a physical serial scanner or a virtual raycasting simulation) must implement.
# It ensures that perception pipelines and navigation controllers receive formatted scans
# uniformly across both offline simulation and real-world testing.
# """

from abc import ABC, abstractmethod
from src.common.types import LidarScan

class LidarDevice(ABC):
    @abstractmethod
    def read_scan(self) -> LidarScan:
        """
        Connect to or query the scanner device, and return a standardized LidarScan.
        """
        pass
