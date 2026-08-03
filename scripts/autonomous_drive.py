# """Autonomous mission execution: running computer vision and machine learning drives.
#
# This script is the central autonomous navigation loop. It boots the vehicle backend,
# initializes safety filters, and runs visual lane line trackers or loads a trained
# Reinforcement Learning policy model to command steering and speed autonomously.
# """

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from src.drive.sim_backend import SimulatedCar
from src.drive.rl_controller import RLCarController
from src.perception.lidar_sim import LidarSimulator
from src.environment.obstacles import ObstacleMap
from src.environment.track import Track
from src.tools.visualizer import TrackVisualizer
from src.rl.agents.random_agent import RandomAgent

def main() -> None:
    """Run autonomous control loop visualizing a RandomAgent driving on track."""
    print("Initializing Autonomous Driving Simulation (RandomAgent)...")

    # 1. Instantiate Track, Physics Car Backend, and Lidar Sensor Simulator
    track = Track(track_name="default_oval", track_width=1.6)
    obs_map = ObstacleMap()
    car = SimulatedCar(wheelbase=0.25, max_speed=5.0)
    car.connect()

    lidar_device = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=3, max_range_m=5.0)

    # 2. Instantiate TrackVisualizer dashboard
    visualizer = TrackVisualizer(track=track, title="Autonomous Driving Simulator - Random Agent Baseline")

    # 3. Create telemetry observer callback to update visualizer frame on each tick
    def update_visualizer(telemetry):
        scan = lidar_device.read_scan()
        frenet = track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)
        visualizer.update(telemetry=telemetry, scan=scan, frenet=frenet)

    # 4. Instantiate RandomAgent policy and RLCarController
    agent = RandomAgent(action_dim=2)
    controller = RLCarController(
        backend=car,
        lidar_dev=lidar_device,
        weights_path="",
        listeners=[update_visualizer],
        agent=agent
    )

    print("Running drive loop... (Close window or wait for steps to complete)")
    dt = 0.05
    steps = 300

    for step in range(steps):
        # Step RL controller tick (queries policy, dispatches command, updates visualizer)
        telemetry = controller.run_step(dt=dt)
        time.sleep(0.02)

        # Check if vehicle drove off-track
        if not track.is_within_boundaries(telemetry.x, telemetry.y):
            print(f"Step {step}: Car drove off-track! Final coordinates: ({telemetry.x:.2f}, {telemetry.y:.2f})")
            break

    print("Autonomous drive simulation complete.")
    time.sleep(1.0)
    visualizer.close()

if __name__ == "__main__":
    main()
