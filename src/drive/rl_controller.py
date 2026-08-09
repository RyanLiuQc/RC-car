# """The RL Inference Controller: driving the vehicle using policy outputs.
#
# This file defines the RLCarController which interfaces a trained PolicyNetwork
# from the src/rl package with the standard CarBackend. It processes real-time
# telemetry data to form observation arrays, executes policy forwards passes to obtain
# control outputs, and dispatches throttle and steering actions directly to the vehicle actuators.
# """

import numpy as np
import time
from typing import List, Callable
from src.common.backend import CarBackend
from src.common.sensor import LidarDevice
from src.common.types import CarTelemetry, CarCommand, LidarScan
from src.rl.base_agent import BaseAgent
from src.rl.agents import * # RandomAgent, A2CAgent, PPOAgent, SACAgent
from src.environment.track import Track

class RLCarController:
    def __init__(
            self, 
            backend: CarBackend, 
            lidar_dev: LidarDevice, 
            weights_path: str, 
            listeners: List[Callable[[CarTelemetry], None]] = None,
            agent: BaseAgent = RandomAgent(),
            track: Track = Track()
                ) -> None:
        
        self.backend: CarBackend = backend
        self.lidar_dev: LidarDevice = lidar_dev
        self.listeners: List[Callable[[CarTelemetry], None]] = list(listeners or [])
        self.track = track
        
        # Instantiate and load trained weights into the policy network
        self.policy: BaseAgent = agent
        if weights_path:
            self.policy.load(weights_path)

    def run_step(self, dt: float = 0.05) -> CarTelemetry:
        """
        Execute a single control tick: query sensors, get policy action, dispatch commands,
        step backend, and notify telemetry listeners.
        """
        telemetry: CarTelemetry = self.backend.telemetry()
        frenet = self.track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)
        
        # 1. Generate Lidar scan polymorphically
        scan: LidarScan = self.lidar_dev.read_scan()
        
        # 2. Assemble 6D normalized observation state [speed, d, heading_err, lidar_r, lidar_c, lidar_l]
        half_width = self.track.track_width / 2.0
        observation: List[float] = [
            telemetry.speed_mps / 5.0,
            frenet.d / half_width,
            frenet.heading_error_deg / 180.0,
            scan.ranges_m[0] / 5.0,
            scan.ranges_m[1] / 5.0,
            scan.ranges_m[2] / 5.0
        ]
        
        # 3. Query policy network
        obs_np = np.array(observation, dtype=np.float32)
        action: np.ndarray = self.policy.select_action(obs_np, deterministic=True)
        cmd = CarCommand(throttle=float(action[0]), steering=float(action[1]))
        
        # 4. Dispatch control signals
        self.backend.send_command(cmd)
        
        # 5. Step physics or pump telemetry comms
        self.backend.update(dt)
        
        # 6. Retrieve fresh telemetry and notify observers
        new_telemetry: CarTelemetry = self.backend.telemetry()
        for listener in self.listeners:
            listener(new_telemetry)
            
        return new_telemetry
