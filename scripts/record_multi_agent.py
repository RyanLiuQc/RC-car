# """Multi-Agent Simulation GIF Recorder: render multiple RL policies simultaneously on one track.
#
# This script loads multiple trained model checkpoints (e.g. SAC v1, v2, v3, v4 or PPO/A2C),
# instantiates independent simulated cars on a shared track, and renders their concurrent
# trajectories to an animated GIF to visualize evolutionary policy progression side-by-side.
#
# Example usage:
#     python -m scripts.record_multi_agent \
#         --weights models/sac/sac_policy_attempt_1.pth \
#                   models/sac/version2/SAC.pth \
#                   models/sac/version3/SAC.pth \
#                   models/sac/version4/SAC.pth \
#         --labels "SAC v1 (Detached Q)" "SAC v2 (Target Lag)" "SAC v3 (Batch Sum Bug)" "SAC v4 (Optimized)" \
#         --output docs/media/sac/sac_progression_comparison.gif \
#         --title "SAC Policy Evolutionary Progression (v1 - v4)"
# """

import os
import sys
import argparse
import math
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drive.sim_backend import SimulatedCar
from src.drive.rl_controller import RLCarController
from src.perception.lidar_sim import LidarSimulator
from src.environment.obstacles import ObstacleMap
from src.environment.track import Track
from src.rl.agents import A2CAgent, PPOAgent, RandomAgent, SACAgent

# Curated high-contrast color palette for distinguishing simultaneous agents
AGENT_COLORS = [
    "#E63946",  # Vibrant Crimson Red
    "#3A86FF",  # Royal Blue
    "#38B000",  # Emerald Green
    "#FF9F1C",  # Amber Orange
    "#9D4EDD",  # Vivid Purple
    "#00F5D4",  # Neon Teal
    "#F72585",  # Deep Pink
    "#FFD166",  # Bright Gold
]

def infer_algo_from_path(weights_path: str) -> str:
    """Infer the RL algorithm from filepath keywords if not explicitly specified."""
    path_lower = weights_path.lower()
    if "sac" in path_lower:
        return "SAC"
    elif "ppo" in path_lower:
        return "PPO"
    elif "a2c" in path_lower:
        return "A2C"
    elif "random" in path_lower:
        return "RANDOM"
    return "SAC"


def create_agent(algo: str):
    """Instantiate an agent instance by algorithm name."""
    if algo == "A2C":
        return A2CAgent(obs_dim=6, action_dim=2)
    elif algo == "PPO":
        return PPOAgent(obs_dim=6, action_dim=2)
    elif algo == "RANDOM":
        return RandomAgent(action_dim=2)
    elif algo == "SAC":
        return SACAgent(obs_dim=6, action_dim=2)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def record_multi_agent_simulation(
    weights_list: List[str],
    labels_list: Optional[List[str]] = None,
    algos_list: Optional[List[str]] = None,
    output_gif_path: str = "docs/media/multi_agent_comparison.gif",
    title: str = "Multi-Agent Policy Simulation Comparison",
    track_name: str = "default_oval",
    num_steps: int = 300,
    frame_skip: int = 2,
    resolution: tuple = (560, 560),
    fps: int = 20
) -> None:
    """Run multiple agents concurrently on a single track and compile an animated comparison GIF."""
    num_agents = len(weights_list)
    if num_agents == 0:
        raise ValueError("At least one weights path must be provided.")

    # Auto-generate labels if omitted
    if not labels_list or len(labels_list) != num_agents:
        labels_list = []
        for p in weights_list:
            base = os.path.splitext(os.path.basename(p))[0]
            parent = os.path.basename(os.path.dirname(p))
            label = f"{parent}/{base}" if parent and parent not in ["models", "."] else base
            labels_list.append(label)

    # Auto-infer algorithms if omitted
    if not algos_list or len(algos_list) != num_agents:
        algos_list = [infer_algo_from_path(p) for p in weights_list]

    print(f"\n=======================================================")
    print(f"Initializing Multi-Agent Simulation Recorder")
    print(f"Track: {track_name} | Steps: {num_steps} | Output: {output_gif_path}")
    print(f"Agents ({num_agents}):")
    for i, (w, lbl, alg) in enumerate(zip(weights_list, labels_list, algos_list)):
        print(f"  [{i+1}] {lbl} ({alg}) -> {w}")
    print(f"=======================================================\n")

    track = Track(track_name=track_name, track_width=1.6)
    obs_map = ObstacleMap()

    # Set up Matplotlib Figure
    plt.ioff()
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=100)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X Position (m)", fontsize=10)
    ax.set_ylabel("Y Position (m)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")

    # 1. Draw Static Track Centerline & Walls
    wx = [w[0] for w in track.waypoints] + [track.waypoints[0][0]]
    wy = [w[1] for w in track.waypoints] + [track.waypoints[0][1]]
    ax.plot(wx, wy, "k--", linewidth=1.0, label="Centerline", alpha=0.35, zorder=1)

    for seg in track.boundaries:
        x1, y1, x2, y2 = seg
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1.8, alpha=0.85, zorder=1)

    # 2. Initialize Agent Data Structures
    agents_data: List[Dict[str, Any]] = []
    for i in range(num_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        weights = weights_list[i]
        algo = algos_list[i]
        label = labels_list[i]

        car = SimulatedCar(wheelbase=0.25, max_speed=5.0)
        car.connect()

        lidar = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=3, max_range_m=5.0)
        agent_obj = create_agent(algo)
        controller = RLCarController(
            backend=car,
            lidar_dev=lidar,
            weights_path=weights,
            listeners=[],
            agent=agent_obj,
            track=track
        )

        # Plot handles for this vehicle
        (traj_line,) = ax.plot([], [], "-", color=color, linewidth=2.0, alpha=0.85, label=label, zorder=3)
        (car_dot,) = ax.plot([], [], "o", color=color, markersize=9, markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        agents_data.append({
            "id": i,
            "label": label,
            "color": color,
            "car": car,
            "controller": controller,
            "traj_line": traj_line,
            "car_dot": car_dot,
            "heading_arrow": None,
            "x_history": [],
            "y_history": [],
            "is_alive": True,
            "crash_step": None,
            "telemetry": car.telemetry(),
        })

    # Status & Legend HUD
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    status_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#CCCCCC")
    )

    frames = []
    dt = 0.05

    # 3. Multi-Agent Simulation Step Loop
    for step in range(1, num_steps + 1):
        hud_lines = [f"Step: {step:3d}/{num_steps} (Time: {step*dt:.1f}s)"]

        for ag in agents_data:
            if ag["is_alive"]:
                # Advance step
                tel = ag["controller"].run_step(dt=dt)
                ag["telemetry"] = tel
                ag["x_history"].append(tel.x)
                ag["y_history"].append(tel.y)

                # Check boundaries
                if not track.is_within_boundaries(tel.x, tel.y):
                    ag["is_alive"] = False
                    ag["crash_step"] = step
                    # Draw a crash marker 'X' at exit location
                    ax.plot([tel.x], [tel.y], "x", color=ag["color"], markersize=10, markeredgewidth=2.0, zorder=6)

            # Update graphical elements
            ag["traj_line"].set_data(ag["x_history"], ag["y_history"])
            if ag["x_history"]:
                ag["car_dot"].set_data([ag["x_history"][-1]], [ag["y_history"][-1]])

            # Update heading arrow
            if ag["heading_arrow"]:
                ag["heading_arrow"].remove()
                ag["heading_arrow"] = None

            if ag["is_alive"]:
                heading_rad = math.radians(ag["telemetry"].heading_deg)
                arrow_dx = 0.45 * math.cos(heading_rad)
                arrow_dy = 0.45 * math.sin(heading_rad)
                ag["heading_arrow"] = ax.arrow(
                    ag["telemetry"].x, ag["telemetry"].y, arrow_dx, arrow_dy,
                    head_width=0.18, head_length=0.18, fc=ag["color"], ec="black", lw=0.5, zorder=5
                )
                hud_lines.append(f"{ag['label'][:20]:<20}: {ag['telemetry'].speed_mps:.2f} m/s")
            else:
                hud_lines.append(f"{ag['label'][:20]:<20}: Crashed (step {ag['crash_step']})")

        status_text.set_text("\n".join(hud_lines))

        # Render frame for GIF
        if step % frame_skip == 0:
            fig.canvas.draw()
            rgba_buf = fig.canvas.buffer_rgba()
            img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba_buf, "raw", "RGBA", 0, 1)
            img_rgb = img.convert("RGB").resize(resolution, Image.Resampling.LANCZOS)
            frames.append(img_rgb)

        # Early termination if all cars have crashed
        if not any(ag["is_alive"] for ag in agents_data):
            print(f"All agents crashed at step {step}. Finishing recording.")
            break

    plt.close(fig)

    # 4. Save Compiled Animated GIF
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    if frames:
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"\nSuccessfully compiled multi-agent comparison GIF: '{output_gif_path}' ({len(frames)} frames, {fps} fps)\n")
    else:
        print("Error: No frames were captured.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simultaneously simulate and record multiple RL policy checkpoints to one animated GIF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to policy checkpoints (.pth) to run concurrently"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Optional custom labels for each car (must match number of weights)"
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="+",
        choices=["A2C", "PPO", "RANDOM", "SAC"],
        help="Optional list of algorithms matching each weight file (auto-inferred if omitted)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/media/multi_agent_progression.gif",
        help="Destination path for output animated GIF"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Multi-Agent Policy Progression Comparison",
        help="Title overlay in top banner"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="default_oval",
        choices=["default_oval", "s_curve", "figure_eight"],
        help="Simulation track layout"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=260,
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Record every N-th simulation step"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output GIF animation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record_multi_agent_simulation(
        weights_list=args.weights,
        labels_list=args.labels,
        algos_list=args.algos,
        output_gif_path=args.output,
        title=args.title,
        track_name=args.track,
        num_steps=args.steps,
        frame_skip=args.frame_skip,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
