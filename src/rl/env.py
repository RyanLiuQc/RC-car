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
from src.environment.track import Track
from src.common.types import CarTelemetry, FrenetState, CarCommand
from src.rl.rewards import RewardCalculator

class RCCarEnv:
    def __init__(self, car: SimulatedCar, lidar_sim: LidarSimulator, track: Track) -> None:
        self.car: SimulatedCar = car
        self.lidar_sim: LidarSimulator = lidar_sim
        self.track: Track = track
        self.rewards: RewardCalculator = RewardCalculator()
        
        # Action space: [throttle, steering]
        # Observation space: [speed, cross_track_error (d), heading_error, lidar_left, lidar_center, lidar_right]
        # TODO: Define formal Gym spaces (Box/Discrete) once Gym is fully integrated.

    def reset(self) -> List[float]:
        """
        Reset the car physics state and simulator to initial track coordinates.
        Returns the initial observation list.
        """
        # TODO: Reset SimulatedCar coordinates and return state observations.
        self.car.connect()
        return [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]

    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """
        Apply control actions, update physical dynamics, and calculate the step metrics.
        Returns a tuple of (observation, reward, terminated, truncated, info).
        """
        throttle: float = action[0]
        steering: float = action[1]
        
        # 1. Apply controls and step the physics backend
        cmd = CarCommand(throttle=throttle, steering=steering)
        self.car.send_command(cmd)
        self.car.update(0.05)
        
        # 2. Query Cartesian telemetry
        telemetry: CarTelemetry = self.car.telemetry()
        
        # 3. Transform Cartesian to track-relative Frenet coordinates
        frenet_state: FrenetState = self.track.cartesian_to_frenet(
            telemetry.x, telemetry.y, telemetry.heading_deg
        )
        
        # 4. Construct observation vector
        # State observations: [speed, lateral_offset (d), heading_error, lidar_left, lidar_center, lidar_right]
        obs: List[float] = [
            telemetry.speed_mps,
            frenet_state.d,
            frenet_state.heading_error_deg
        ] + telemetry.lidar_scan.ranges_m
        
        # 5. Compute step reward using the Frenet state
        # TODO: Track width should be queried from track configuration
        track_width = 1.6
        is_off_track: bool = abs(frenet_state.d) > (track_width / 2.0)
        terminated: bool = telemetry.crashed or is_off_track
        truncated: bool = False
        
        reward: float = self.rewards.compute_reward(telemetry, frenet_state)
        
        info: Dict[str, Any] = {
            "s": frenet_state.s,
            "d": frenet_state.d,
            "heading_error": frenet_state.heading_error_deg
        }
        
        return obs, reward, terminated, truncated, info
