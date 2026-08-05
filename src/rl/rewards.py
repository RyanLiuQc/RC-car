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
        
        # get reward if target velocity ONLY IF lateral dev is small
        speed_reward = max(0.0,(1.0 - abs(telemetry.speed_mps - self.target_speed)/self.target_speed))

        # cos(heading_error) (speed_reward * aligment)
        velocity_reward =  max(0.0, speed_reward * math.cos(math.radians(frenet_state.heading_error_deg)))

        # clamp to prevent less negative rewarding when heading_err>90 deg when d is small
        # Exponential lateral dev penalty scales to eliminate wall hugging that linear penalty couldn't
        vel_and_lateral_penalty = max(0.0, velocity_reward * math.e**(-(frenet_state.d/self.std)**2))

        # Compute progress rate per step:
        delta_s = max(0.0, frenet_state.s - self.prev_s)
        progress_reward = delta_s / dt  # m/s along centerline
        # Store current s for next step:
        self.prev_s = frenet_state.s

        return  vel_and_lateral_penalty + progress_reward
        
        # # Calculate speed_reward (penalize difference between target and actual speed)
        # speed_reward = max(0.0, 1.0 - abs(telemetry.speed_mps - self.target_speed))
        # # adding directionality to the speed reward
        # # velocity_reward = max(-1.0, (1.0 - abs(telemetry.speed_mps - self.target_speed))*math.cos(frenet_state.heading_error_deg))

        # # Calculate lateral_penalty (penalize large d)
        # lateral_penalty = 0.5 * abs(frenet_state.d)

        # # Calculate heading_penalty = 0.01 * abs(frenet_state.heading_error_deg)
        # heading_penalty = 0.01 * abs(frenet_state.heading_error_deg)

        # # Return total reward = speed_reward - lateral_penalty - heading_penalty
        # reward = speed_reward - lateral_penalty - heading_penalty

        # return reward
        
        
