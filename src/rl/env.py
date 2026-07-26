# """The Gym environment wrapper: standardizing observations, actions, and steps.
#
# This file adapts the simulated car kinematics (from src/drive/sim_backend) and
# the Lidar simulated sensor ranges (from src/perception/lidar_sim) into a unified,
# standard Gymnasium interface. It enables off-the-shelf reinforcement learning algorithms
# (such as PPO or SAC) to interact seamlessly with the vehicle dynamics.
# """
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from typing import Tuple, Dict, Any, List
from src.drive.sim_backend import SimulatedCar
from src.perception.lidar_sim import LidarSimulator
from src.environment.track import Track
from src.common.types import CarTelemetry, FrenetState, CarCommand
from src.rl.rewards import RewardCalculator

from src.tools.visualizer import TrackVisualizer

class RCCarEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
            self, car: SimulatedCar, 
            lidar_sim: LidarSimulator, 
            track: Track, 
            render_mode: str = None
            ) -> None:
        
        self.car: SimulatedCar = car
        self.lidar_sim: LidarSimulator = lidar_sim
        self.track: Track = track
        self.render_mode = render_mode
        self.rewards: RewardCalculator = RewardCalculator()

        self.visualizer = None

        # Define formal Gym spaces (Box/Discrete)
        # refer to: https://gymnasium.farama.org/api/spaces/fundamental/
        # Action space: [throttle, steering] (action[0] := throttle, action[1] := steering)
        self.action_space: spaces.Space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), # low throttle and low steering
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space: [speed, cross_track_error (d), heading_error, lidar_right, lidar_center, lidar_left]
        low_obs = np.array([0.0, -self.track.track_width/2-1, -180, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([self.car.max_speed, -self.track.track_width/2+1, 180, 5.0, 5.0, 5.0], dtype=np.float32)
        observation_space: spaces.Space = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(6,),
            dtype=np.float32
        )

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the car physics state and simulator to initial track coordinates.
        Returns the initial observation list.
        """
        super().reset(seed=seed)

        # Reset car position to start line (0,0) and zero speed
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0       # Radians
        self.speed = 0.0         # m/s
        self.steering = 0.0      # Radians
        self.battery = 100.0
        self.car.connect()

        self.car.send_command(CarCommand(throttle=0.0, steering=0.0, brake=0.0))

        # Query initial telemetry and lidar scan
        telemetry = self.car.telemetry()
        lidar_scan = self.lidar_sim.read_scan

        # Construct initial observation vector

        # Return (initial_obs, info_dict)
        
        return [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply control actions, update physical dynamics, and calculate the step metrics.
        Returns a tuple of (observation, reward, terminated, truncated, info).
        """

        #  Unpack throttle and steering from action
        #  Issue CarCommand to self.car and call self.car.update(dt=0.05)
        #  Query frenet = self.track.cartesian_to_frenet(...)
        telemetry = self.car.telemetry()
        frenet = self.track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)
        #  Construct observation vector [speed, d, heading_err, lidar_r, lidar_c, lidar_l]
        lidar_scan = self.lidar_sim.read_scan()

        #  Compute reward = self.rewards.compute_reward(telemetry, frenet)
        #  Determine terminated flag (crashed or off-track)
        #  If self.render_mode == "human", trigger rendering visualization
        #  Return (obs, reward, terminated, truncated, info)



        if self.visualizer and self.render_mode == "human":
            self.visualizer.update(telemetry=telemetry, scan=lidar_scan, frenet=frenet)

        # throttle: float = action[0]
        # steering: float = action[1]
        
        # # 1. Apply controls and step the physics backend
        # cmd = CarCommand(throttle=throttle, steering=steering)
        # self.car.send_command(cmd)
        # self.car.update(0.05)
        
        # # 2. Query Cartesian telemetry
        # telemetry: CarTelemetry = self.car.telemetry()
        
        # # 3. Transform Cartesian to track-relative Frenet coordinates
        # frenet_state: FrenetState = self.track.cartesian_to_frenet(
        #     telemetry.x, telemetry.y, telemetry.heading_deg
        # )
        
        # # 4. Construct observation vector
        # # State observations: [speed, lateral_offset (d), heading_error, lidar_left, lidar_center, lidar_right]
        # obs: List[float] = [
        #     telemetry.speed_mps,
        #     frenet_state.d,
        #     frenet_state.heading_error_deg
        # ] + telemetry.lidar_scan.ranges_m
        
        # # 5. Compute step reward using the Frenet state
        # # TODO: Track width should be queried from track configuration
        # track_width = 1.6
        # is_off_track: bool = abs(frenet_state.d) > (track_width / 2.0)
        # terminated: bool = telemetry.crashed or is_off_track
        # truncated: bool = False
        
        # reward: float = self.rewards.compute_reward(telemetry, frenet_state)
        
        # info: Dict[str, Any] = {
        #     "s": frenet_state.s,
        #     "d": frenet_state.d,
        #     "heading_error": frenet_state.heading_error_deg
        # }
        
        # return obs, reward, terminated, truncated, info


    def render(self) -> None:
            """
            Render 2D visualization if render_mode is 'human'.
            """
            if self.render_mode == "human":
                # Plot car position (x, y) on track window
                self.visualizer = TrackVisualizer(title="RC Car 2D Kinematic Simulator for rl")
                
if __name__ == "__main__":
    import gymnasium as gym

    # Create a classic CartPole environment
    env = gym.make("CartPole-v1", render_mode="human")

    # Reset the environment to start
    observation, info = env.reset(seed=42)

    for _ in range(100):
        # Take a random action
        action = env.action_space.sample() 
        
        # Step through the simulation
        observation, reward, terminated, truncated, info = env.step(action)

        # Reset if the game ends
        if terminated or truncated:
            observation, info = env.reset()

    env.close()