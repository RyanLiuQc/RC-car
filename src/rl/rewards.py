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

    def compute_reward(self, telemetry: CarTelemetry, frenet_state: FrenetState) -> float:
        """
        Calculate step-level reward based on speed error, lateral track displacement d, and crash status.
        """
        # Terminal penalty for crashes or driving far off centerline
        # TODO: Read track width constraints
        track_width = 1.6
        is_off_track = abs(frenet_state.d) > (track_width / 2.0)
        if telemetry.crashed or is_off_track:
            return -100.0
        
        # Reward maintaining cruise speed
        speed_reward: float = 1.0 - abs(telemetry.speed_mps - self.target_speed)
        
        # Penalize deviation from the centerline
        lateral_penalty: float = abs(frenet_state.d) * 0.5
        
        return max(0.0, speed_reward - lateral_penalty)
