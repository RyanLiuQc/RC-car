# """The Gym environment wrapper: standardizing observations, actions, and steps.
#
# This file adapts the simulated car kinematics (from src/drive/sim_backend) and
# the Lidar simulated sensor ranges (from src/perception/lidar_sim) into a unified,
# standard Gymnasium interface. It enables off-the-shelf reinforcement learning algorithms
# (such as PPO or SAC) to interact seamlessly with the vehicle dynamics.
# """

from typing import Tuple, Dict, Any, List
from src.drive.sim_backend import SimulatedCar
from src.perception.lidar_sim import LidarSimulator

class RCCarEnv:
    def __init__(self, car: SimulatedCar, lidar_sim: LidarSimulator) -> None:
        self.car: SimulatedCar = car
        self.lidar_sim: LidarSimulator = lidar_sim
        
        # Action space: [throttle, steering]
        # Observation space: [speed, lidar_left, lidar_center, lidar_right]
        # TODO: Define formal Gym spaces (Box/Discrete) once Gym is fully integrated.

    def reset(self) -> List[float]:
        """
        Reset the car physics state and simulator to initial track coordinates.
        Returns the initial observation list.
        """
        # TODO: Reset SimulatedCar coordinates and return state observations.
        self.car.connect()
        return [0.0, 5.0, 5.0, 5.0]

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """
        Apply control actions, update physical dynamics, and calculate the step metrics.
        Returns a tuple of (observation, reward, terminated, truncated, info).
        """
        # TODO: Apply controls, update backend, compute reward from rewards.py, and detect collisions.
        throttle: float = action[0]
        steering: float = action[1]
        self.car.set_controls(throttle, steering)
        self.car.update(0.05)
        
        obs: List[float] = [self.car.speed, 5.0, 5.0, 5.0]
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False
        info: Dict[str, Any] = {}
        
        return obs, reward, terminated, truncated, info
