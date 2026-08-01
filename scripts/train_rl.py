# """RL Training Execution Script: command-line entrypoint for training policies.
#
# This script instantiates the Gymnasium environment (RCCarEnv), parses command-line
# arguments (--algo RANDOM/PPO/SAC, --timesteps, --visualize), configures render modes,
# and runs the policy training loop.
# """

import argparse
from src.rl.agents import RandomAgent, PPOAgent, SACAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous RC Car RL Training Script")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="RANDOM", 
        choices=["RANDOM", "PPO", "SAC"],
        help="Select RL training algorithm (default: RANDOM)"
    )
    parser.add_argument("--timesteps", type=int, default=1000, help="Total training timesteps")
    parser.add_argument("--visualize", action="store_true", help="Enable live 2D visual rendering")
    return parser.parse_args()

def main():
    # Instantiate Track, SimulatedCar, LidarSimulator, and RCCarEnv(..., render_mode=render_mode)
    # Instantiate RL model (e.g. PPO("MlpPolicy", env, verbose=1))
    # Train policy: model.learn(total_timesteps=10000)
    # Save trained policy model weights: model.save("rc_car_ppo_policy")

    args = parse_args()
    render_mode = "human" if args.visualize else None
    print(f"Initializing Training Pipeline with Algorithm: {args.algo} ({'Visualized' if args.visualize else 'Headless'} mode)")

    # 1. Instantiate Agent based on CLI argument
    if args.algo == "RANDOM":
        agent = RandomAgent(action_dim=2)
    elif args.algo == "PPO":
        agent = PPOAgent(obs_dim=6, action_dim=2, lr=3e-4)
    elif args.algo == "SAC":
        agent = SACAgent(obs_dim=6, action_dim=2, lr=3e-4)

if __name__ == "__main__":
    main()
