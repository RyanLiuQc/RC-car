# """The reward function shaping module: guiding the policy to drive optimally.
#
# This file contains functions that calculate step-level rewards and penalties during training.
# It implements incentives for maintaining a target cruise speed and keeping the vehicle centered,
# and applies steep penalties when collision boundaries or static obstacles are struck.
# """

from src.common.types import CarTelemetry, FrenetState
import math
import numpy as np

class RewardCalculator:
    def __init__(self, target_speed: float = 1.5, half_track_width: float = 0.8) -> None:
        self.target_speed: float = target_speed

        # Standard deviation set to 30% of half-width (0.24m)
        self.std = 0.3 * half_track_width

        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.prev_action = np.zeros(2, dtype=np.float32)

    def compute_reward( # usually input = obs, reward, next_obs
            self, 
            telemetry: CarTelemetry, 
            frenet_state: FrenetState,
            action_np: np.ndarray = None,
            dt: float = 0.05,
            crashed: bool = False) -> float:
        """
        Calculate step-level reward based on speed error, lateral track displacement d, and crash status.
        """
        # Penalize crash
        if crashed:
            return -100
        
        # Normalized velocity progress along centerline [0, 1]
        progress_reward = telemetry.speed_mps/self.target_speed * math.cos(math.radians(frenet_state.heading_error_deg))
        progress_reward = max(0.0, progress_reward)

        # Centering penalty (Gaussian decay from centerline)
        centering_factor = math.exp(-(frenet_state.d/self.std) ** 2)


        # Steering rate penalty to prevent high-frequency oscillations ("jerk")
        if action_np is not None:   
            # Penalize squared changes (high penalty for sharp/abrupt snaps)
            delta_throttle = (action_np[0] - self.prev_action[0])**2
            delta_steering = (action_np[1] - self.prev_action[1])**2 

            jerk_penalty = 0.05 * delta_steering + 0.1 * delta_throttle

            # Update prev_action memory for next step
            self.prev_action = action_np.copy()
        else:
            jerk_penalty = 0

        total_reward = (0.5 * progress_reward * centering_factor) + (0.5 * centering_factor) - jerk_penalty
        
        return total_reward
