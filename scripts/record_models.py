# """Script to record RL policy simulation runs to animated GIFs for documentation."""

import os
import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering for server/script environments
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drive.sim_backend import SimulatedCar
from src.drive.rl_controller import RLCarController
from src.perception.lidar_sim import LidarSimulator
from src.environment.obstacles import ObstacleMap
from src.environment.track import Track
from src.tools.visualizer import TrackVisualizer
from src.rl.agents import A2CAgent, PPOAgent, RandomAgent

def record_policy(
    algo: str,
    weights_path: str,
    output_gif_path: str,
    title: str,
    track_name: str = "default_oval",
    num_steps: int = 260,
    frame_skip: int = 2
) -> None:
    """Record a policy drive simulation into an animated GIF."""
    print(f"Recording policy [{algo}] simulation: '{weights_path}' -> '{output_gif_path}'...")

    track = Track(track_name=track_name, track_width=1.6)
    obs_map = ObstacleMap()
    car = SimulatedCar(wheelbase=0.25, max_speed=5.0)
    car.connect()

    lidar_device = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=3, max_range_m=5.0)
    visualizer = TrackVisualizer(track=track, title=title)

    frames = []

    # Instantiate Agent based on algorithm type
    if algo == "A2C":
        agent = A2CAgent(obs_dim=6, action_dim=2)
    elif algo == "PPO":
        agent = PPOAgent(obs_dim=6, action_dim=2)
    elif algo == "RANDOM":
        agent = RandomAgent(action_dim=2)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    controller = RLCarController(
        backend=car,
        lidar_dev=lidar_device,
        weights_path=weights_path,
        listeners=[],
        agent=agent,
        track=track
    )

    dt = 0.05
    for step in range(1, num_steps + 1):
        telemetry = controller.run_step(dt=dt)
        scan = lidar_device.read_scan()
        frenet = track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)

        visualizer.update(telemetry=telemetry, scan=scan, frenet=frenet)

        if step % frame_skip == 0:
            visualizer.fig.canvas.draw()
            rgba_buf = visualizer.fig.canvas.buffer_rgba()
            img = Image.frombuffer("RGBA", visualizer.fig.canvas.get_width_height(), rgba_buf, "raw", "RGBA", 0, 1)
            img_rgb = img.convert("RGB").resize((480, 480), Image.Resampling.LANCZOS)
            frames.append(img_rgb)

        if not track.is_within_boundaries(telemetry.x, telemetry.y):
            print(f"Step {step}: Car went off-track!")
            break

    visualizer.close()

    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # 50ms per frame = 20 fps
            loop=0
        )
        print(f"Successfully saved recording to {output_gif_path} ({len(frames)} frames)")

def parse_args():
    parser = argparse.ArgumentParser(description="Record RL policy simulation to animated GIF")
    parser.add_argument("--algo", type=str, default="PPO", choices=["A2C", "PPO", "RANDOM"], help="Policy algorithm")
    parser.add_argument("--weights", type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, help="Output GIF path")
    parser.add_argument("--title", type=str, help="Title overlay for visualization")
    parser.add_argument("--track", type=str, default="default_oval", choices=["default_oval", "s_curve", "figure_eight"])
    parser.add_argument("--steps", type=int, default=260, help="Simulation steps to record")
    return parser.parse_args()

def record_ppo_attempts():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(repo_root, "models")
    media_dir = os.path.join(repo_root, "docs", "media")

    ppo_models = [
        ("PPO", os.path.join(models_dir, "PPO_policy_exploites_small_d_by_rotating.pth"), os.path.join(media_dir, "ppo_policy_rotation_exploit.gif"), "PPO Attempt 1 (Rotation Exploit)"),
        ("PPO", os.path.join(models_dir, "PPO_policy_update_reward.pth"), os.path.join(media_dir, "ppo_policy_update_reward.gif"), "PPO Attempt 2 (Centerline Lock & Oscillations)"),
        ("PPO", os.path.join(models_dir, "PPO_policy_update_reward_1.pth"), os.path.join(media_dir, "ppo_policy_update_reward_1.gif"), "PPO Attempt 3 (Smooth Lap Completion)"),
    ]

    for algo, weights, gif_path, title in ppo_models:
        if os.path.exists(weights):
            record_policy(algo, weights, gif_path, title=title, num_steps=260)
        else:
            print(f"Warning: Checkpoint '{weights}' not found, skipping...")

def main():
    args = parse_args()
    if args.weights:
        output_path = args.output or f"docs/media/{os.path.splitext(os.path.basename(args.weights))[0]}.gif"
        title = args.title or f"{args.algo} Simulation ({os.path.basename(args.weights)})"
        record_policy(args.algo, args.weights, output_path, title=title, track_name=args.track, num_steps=args.steps)
    else:
        record_ppo_attempts()

if __name__ == "__main__":
    main()
