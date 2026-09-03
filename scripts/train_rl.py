# """RL Training Execution Script: command-line entrypoint for training policies.
#
# This script instantiates the Gymnasium environment (RCCarEnv), parses command-line
# arguments (--algo RANDOM/PPO/SAC/A2C, --timesteps, --visualize), configures render modes,
# and runs the policy training loop.
# Ex:  python -m scripts.train_rl --algo SAC
# """

import argparse
from src.rl.agents import * # RandomAgent, PPOAgent, SACAgent, A2CAgent
from src.rl.env import RCCarEnv
from src.environment.track import Track
from src.environment.obstacles import ObstacleMap
from src.perception.lidar_sim import LidarSimulator
from src.drive.sim_backend import SimulatedCar


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous RC Car RL Training Script")
    parser.add_argument(
        "--algo", 
        type=str, 
        default="PPO", 
        choices=["RANDOM", "PPO", "SAC", "A2C"],
        help="Select RL training algorithm (default: RANDOM)"
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--visualize", action="store_true", help="Enable live 2D visual rendering")
    parser.add_argument("--path", type=str, help="Weight's path")
    parser.add_argument("--load-weights", type=str, help="Path to pre-trained model weights file to load before training")

    return parser.parse_args()

def main():
    # Train policy: model.learn(total_timesteps=10000) # args.timesteps?
    # Save trained policy model weights: model.save("rc_car_ppo_policy")

    args = parse_args()
    render_mode = "human" if args.visualize else None
    print(f"Initializing Training Pipeline with Algorithm: {args.algo} ({'Visualized' if args.visualize else 'Headless'} mode)")

    # 1. Instantiate Agent based on CLI argument
    if args.algo == "RANDOM":
        agent = RandomAgent(action_dim=2)
        weights_path = "models/random_policy.pth"
    elif args.algo == "PPO":
        agent = PPOAgent(obs_dim=6, action_dim=2, device="cpu")
        weights_path = "models/ppo_policy.pth"
    elif args.algo == "SAC":
        agent = SACAgent(obs_dim=6, action_dim=2)
        weights_path = "models/sac/sac_policy.pth"
    elif args.algo == "A2C":
        agent = A2CAgent(obs_dim=6, action_dim=2, actor_lr=1e-4, critic_lr=3e-4)
        weights_path = "models/a2c_policy_3.pth"
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Load pre-trained model weights if specified
    if args.load_weights:
        print(f"Loading pre-trained weights from: {args.load_weights}")
        agent.load(args.load_weights)

    # Instantiate Track, SimulatedCar, LidarSimulator, and RCCarEnv(..., render_mode=render_mode)
    track = Track()
    obstacles = ObstacleMap()
    car = SimulatedCar()
    lidar_dev = LidarSimulator(obstacles, car, track)
    env = RCCarEnv(car, lidar_dev, track, obstacles, render_mode=render_mode)

    # main loop
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    
    print(f"Starting Training for {args.timesteps} timesteps...")

    for step in range(1, args.timesteps + 1):
        # select an action using the agent's current policy
        action = agent.select_action(obs)

        # step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        episode_reward += reward

        # Build trajectory buffer dict for agent update
        trajectory_buffer: dict = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs, # after action was taken
            "terminated": terminated, # if next_obs hits obstacle or wall, sets next_value to 0 (reset positiona and state value)
            "done": done # for cutoff gae
            # truncated: bootstrap next_value = critic(next_obs) (it's still alive, so estimate its future value).
        }


        metrics = agent.train_step(trajectory_buffer=trajectory_buffer, step=step)

        if (step % 500 == 0 and metrics) and ((args.algo == "PPO" or args.algo == "SAC") and metrics):
            print(f"[Step {step}/{args.timesteps}] Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f} | Entropy: {metrics['entropy']:.2f}")

        if done: 
            # update num of episode, reset env, reset episode reward.
            episode_count += 1
            print(f"Episode {episode_count} | Steps: {step} | Total Reward: {episode_reward:.2f}")

            obs, info = env.reset()
            episode_reward = 0.0

        else:
            obs = next_obs

    # save weights
    #agent.save(weights_path)
    agent.save(args.path or weights_path)

    
    

if __name__ == "__main__":
    main()
