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
from src.environment.obstacles import ObstacleMap
from src.common.types import CarTelemetry, FrenetState, CarCommand
from src.rl.rewards import RewardCalculator

from src.tools.visualizer import TrackVisualizer

class RCCarEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
            self, car: SimulatedCar, 
            lidar_sim: LidarSimulator, 
            track: Track, 
            obstacle_map: ObstacleMap = None,
            render_mode: str = None,
            max_episode_steps: int = 1000
            ) -> None:
        
        self.car: SimulatedCar = car
        self.lidar_sim: LidarSimulator = lidar_sim
        self.track: Track = track
        self.obstacle_map = obstacle_map
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
        low_obs = np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space: spaces.Space = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(6,),
            dtype=np.float32
        )

        # Step count
        self.current_step = 0
        self.max_episode_steps = max_episode_steps

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the car physics state and simulator to initial track coordinates.
        Returns the initial observation list.
        """
        self.current_step = 0
        super().reset(seed=seed)

        # Reset car position to start line (0,0) and zero speed
        self.car.t = 0.0
        self.car.x = 0.0
        self.car.y = 0.0
        self.car.heading = 0.0       # Radians
        self.car.speed = 0.0         # m/s
        self.car.steering = 0.0      # Radians
        self.car.battery = 100.0
        self.car.connect()

        self.car.send_command(CarCommand(throttle=0.0, steering=0.0, brake=0.0))

        # Query initial telemetry and lidar scan
        telemetry = self.car.telemetry()
        lidar_scan = self.lidar_sim.read_scan()

        # Construct initial observation vector
        frenet = self.track.cartesian_to_frenet(self.car.x, self.car.y, np.degrees(self.car.heading))
        half_width = self.track.track_width / 2.0

        obs = np.array([ 
            telemetry.speed_mps / 5.0,          # Normalized
            frenet.d / half_width,
            frenet.heading_error_deg / 180.0,
            lidar_scan.ranges_m[0] / 5.0,
            lidar_scan.ranges_m[1] / 5.0,
            lidar_scan.ranges_m[2] / 5.0
        ], dtype=np.float32)

        # Return (initial_obs, info_dict)
        info = {}
        return obs, info
    
    def _check_crash(self, telemetry: CarTelemetry, frenet: FrenetState) -> bool:
        """Check if vehicle has driven off-track or hit an obstacle."""
        is_off_track = abs(frenet.d) > self.track.track_width / 2.0
        is_obstacle_hit = self.obstacle_map and self.obstacle_map.is_obstacle(telemetry.x,telemetry.y)
            
        return is_off_track or is_obstacle_hit

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply control actions, update physical dynamics, and calculate the step metrics.
        Returns a tuple of (observation, reward, terminated, truncated, info).
        """
        self.current_step += 1

        #  Unpack throttle and steering from action
        throttle = action[0]
        steering = action[1]

        #  Issue CarCommand to self.car and call self.car.update(dt=0.05)
        command = CarCommand(throttle=throttle, steering=steering)
        self.car.send_command(command)
        self.car.update(dt=0.05)

        #  Query frenet = self.track.cartesian_to_frenet(...)
        telemetry: CarTelemetry = self.car.telemetry()
        frenet: FrenetState = self.track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)

        #  Construct observation vector [speed, d, heading_err, lidar_r, lidar_c, lidar_l]
        lidar_scan = self.lidar_sim.read_scan()
        half_width = self.track.track_width / 2.0
        obs = np.array([                
            telemetry.speed_mps / 5.0,      # Normalized
            frenet.d / half_width, 
            frenet.heading_error_deg / 180,
            lidar_scan.ranges_m[0] / 5.0,
            lidar_scan.ranges_m[1] / 5.0,
            lidar_scan.ranges_m[2] / 5.0
            ], dtype=np.float32)

        #  Determine terminated flag (crashed or off-track)
        terminated = self._check_crash(telemetry, frenet)

        #  Compute reward = self.rewards.compute_reward(telemetry, frenet)
        reward = self.rewards.compute_reward(telemetry=telemetry, frenet_state=frenet, crashed=terminated)


        #  If self.render_mode == "human", trigger rendering visualization
        if self.visualizer and self.render_mode == "human":
            self.visualizer.update(telemetry=telemetry, scan=lidar_scan, frenet=frenet)

        # Truncated: TODO: set limit of episodes. add to init and keep count of steps
        truncated = self.current_step >= self.max_episode_steps

        # info
        info = {
            "s": frenet.s,                             # Total distance traveled along centerline (m)
            "d": frenet.d,                             # Raw lateral offset (m)
            "heading_error_deg": frenet.heading_error_deg, # Raw yaw error (deg)
            "speed_mps": telemetry.speed_mps,          # Raw speed (m/s)
            "is_crashed": terminated,                  # Crash status
            "step": self.current_step                  # Step count
        }


        #  Return (obs, reward, terminated, truncated, info)
        return (obs, reward, terminated, truncated, info)


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