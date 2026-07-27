# """The reward function shaping module: guiding the policy to drive optimally.
#
# This file contains functions that calculate step-level rewards and penalties during training.
# It implements incentives for maintaining a target cruise speed and keeping the vehicle centered,
# and applies steep penalties when collision boundaries or static obstacles are struck.
# """

from src.common.types import CarTelemetry, FrenetState

class RewardCalculator:
    def __init__(self, target_speed: float = 1.5) -> None:
        self.target_speed: float = target_speed

    def compute_reward(
            self, 
            telemetry: CarTelemetry, 
            frenet_state: FrenetState,
            crashed: bool = False) -> float:
        """
        Calculate step-level reward based on speed error, lateral track displacement d, and crash status.
        """
        # Penalize crash
        if crashed:
            return -100
        
        # Calculate speed_reward (penalize difference between target and actual speed)
        speed_reward = max(0.0, 1.0 - abs(telemetry.speed_mps - self.target_speed))

        # Calculate lateral_penalty (penalize large d)
        lateral_penalty = 0.5 * abs(frenet_state.d)

        # Calculate heading_penalty = 0.01 * abs(frenet_state.heading_error_deg)
        heading_penalty = 0.01 * abs(frenet_state.heading_error_deg)

        # Return total reward = speed_reward - lateral_penalty - heading_penalty
        reward = speed_reward - lateral_penalty - heading_penalty

        return reward
        
        
