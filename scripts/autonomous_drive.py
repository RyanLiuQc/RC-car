# """Autonomous mission execution: running computer vision and machine learning drives.
#
# This script is the central autonomous navigation loop. It boots the vehicle backend,
# initializes safety filters, and runs visual lane line trackers or loads a trained
# Reinforcement Learning policy model to command steering and speed autonomously.
# """

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import numpy as np
from src.drive.sim_backend import SimulatedCar
from src.drive.rl_controller import RLCarController
from src.perception.lidar_sim import LidarSimulator
from src.environment.obstacles import ObstacleMap
from src.environment.track import Track
from src.tools.visualizer import TrackVisualizer
from src.rl.agents import RandomAgent, A2CAgent, PPOAgent, SACAgent

def parse_args():
    """Parse command-line arguments for autonomous driving visualization."""
    parser = argparse.ArgumentParser(description="Autonomous RC Car RL Inference & Visualization Script")
    parser.add_argument(
        "--algo",
        type=str,
        default="A2C",
        choices=["RANDOM", "A2C", "PPO", "SAC"],
        help="Select RL policy algorithm (default: A2C)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/a2c_policy_new_reward.pth",
        help="Path to trained model weights checkpoint (.pth)"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="default_oval",
        choices=["default_oval", "s_curve", "figure_eight"],
        help="Select track layout (default: default_oval)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum simulation drive steps (default: 1000)"
    )
    return parser.parse_args()

def main() -> None:
    """Run autonomous control loop visualizing selected agent policy on track."""
    args = parse_args()

    print(f"Initializing Autonomous Driving Simulation...")
    print(f"  - Algorithm: {args.algo}")
    print(f"  - Weights Path: {args.weights if args.algo != 'RANDOM' else 'N/A (Random Policy)'}")
    print(f"  - Track Layout: {args.track}")

    # 1. Instantiate Track, Physics Car Backend, and Lidar Sensor Simulator
    track = Track(track_name=args.track, track_width=1.6)
    obs_map = ObstacleMap()
    car = SimulatedCar(wheelbase=0.25, max_speed=5.0)
    car.connect()

    lidar_device = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=3, max_range_m=5.0)

    # 2. Instantiate TrackVisualizer dashboard
    visualizer = TrackVisualizer(track=track, title=f"Autonomous Driving - {args.algo} ({args.track})")

    # 3. Create telemetry observer callback to update visualizer frame on each tick
    def update_visualizer(telemetry):
        scan = lidar_device.read_scan()
        frenet = track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)
        visualizer.update(telemetry=telemetry, scan=scan, frenet=frenet)

    # 4. Instantiate Agent based on CLI arguments
    if args.algo == "RANDOM":
        agent = RandomAgent(action_dim=2)
        weights_path = ""
    elif args.algo == "A2C":
        agent = A2CAgent(obs_dim=6, action_dim=2)
        weights_path = args.weights
    elif args.algo == "PPO":
        agent = PPOAgent(obs_dim=6, action_dim=2)
        weights_path = args.weights
    elif args.algo == "SAC":
        agent = SACAgent(obs_dim=6, action_dim=2)
        weights_path = args.weights
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # 5. Instantiate RLCarController with loaded policy
    controller = RLCarController(
        backend=car,
        lidar_dev=lidar_device,
        weights_path=weights_path,
        listeners=[update_visualizer],
        agent=agent,
        track=track
    )

    print(f"Running drive loop for up to {args.steps} steps... (Close window to exit)")
    dt = 0.05

    for step in range(1, args.steps + 1):
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
