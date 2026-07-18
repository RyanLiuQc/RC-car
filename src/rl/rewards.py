# """The reward function shaping module: guiding the policy to drive optimally.
#
# This file contains functions that calculate step-level rewards and penalties during training.
# It implements incentives for maintaining a target cruise speed and keeping the vehicle centered,
# and applies steep penalties when collision boundaries or static obstacles are struck.
# """

from src.common.types import CarTelemetry

class RewardCalculator:
    def __init__(self, target_speed: float = 1.5) -> None:
        self.target_speed: float = target_speed

    def compute_reward(self, telemetry: CarTelemetry, is_off_track: bool) -> float:
        """
        Calculate step-level reward based on speed error, steering stability, and crash status.
        """
        # TODO: Compute progress reward, deviation penalty, and terminal crash/estop cost.
        if telemetry.crashed or is_off_track:
            return -100.0
        
        # Reward maintaining cruise speed
        speed_reward: float = 1.0 - abs(telemetry.speed_mps - self.target_speed)
        return max(0.0, speed_reward)
