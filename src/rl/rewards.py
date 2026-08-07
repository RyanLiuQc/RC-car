# """The reward function shaping module: guiding the policy to drive optimally.
#
# This file contains functions that calculate step-level rewards and penalties during training.
# It implements incentives for maintaining a target cruise speed and keeping the vehicle centered,
# and applies steep penalties when collision boundaries or static obstacles are struck.
# """

from src.common.types import CarTelemetry, FrenetState
import math

class RewardCalculator:
    def __init__(self, target_speed: float = 1.5, half_track_width: float = 0.8) -> None:
        self.target_speed: float = target_speed

        # Standard deviation set to 30% of half-width (0.24m)
        self.std = 0.3 * half_track_width

        self.prev_s = 0

    def reset(self) -> None:
        self.prev_s = 0.0

    def compute_reward(
            self, 
            telemetry: CarTelemetry, 
            frenet_state: FrenetState,
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

        # TODO: integrate previous action as in the state of the 
        # Steering rate penalty to prevent high-frequency oscillations ("jerk")
        # steering_jerk_penalty = 0.05 * ((action[1] - prev_action[1]) ** 2)

        total_reward = (0.5 * progress_reward * centering_factor) + (0.5 * centering_factor) # - steering_jerk_penalty
        
        return total_reward
