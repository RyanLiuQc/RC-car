import argparse

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate RL Car Agent")
    parser.add_argument("--visualize", action="store_true", help="Enable live 2D visual rendering during evaluation")
    args = parser.parse_args()

    render_mode = "human" if args.visualize else None
    
    # Instantiate Track, SimulatedCar, LidarSimulator, and RCCarEnv(..., render_mode=render_mode)
    # Instantiate RL model (e.g. PPO("MlpPolicy", env, verbose=1))
    # Train policy: model.learn(total_timesteps=10000)
    # Save trained policy model weights: model.save("rc_car_ppo_policy")
    print(f"Training initialized in {'Visualized' if args.visualize else 'Headless'} mode...")

if __name__ == "__main__":
    main()
