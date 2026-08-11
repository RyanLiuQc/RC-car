# Autonomous RC-Car Control Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

## Overview

This repository contains a modular, clean, and highly decoupled software platform designed for autonomous driving development. It provides a structured foundation that enables a seamless transition between a 2D Kinematic Simulation environment and a physical, servo-driven 1/8 scale RC car chassis. 

By enforcing strict boundaries between physical dynamics, track layout geometry, sensor perception systems, and control logic, the platform supports rapid prototyping of classical controllers, computer vision pipelines, and Reinforcement Learning policies.

## Policy Generalization Across Track Layouts

Demonstration of our top-performing policy (**PPO Attempt 3**, `PPO_policy_update_reward_1.pth`) evaluated zero-shot across simulation track layouts without retraining:

| `default_oval` Track | `s_curve` Track |
| :---: | :---: |
| <img src="docs/media/ppo_policy_update_reward_1.gif" width="280" /> | <img src="docs/media/ppo_policy_update_reward_1_s_curve.gif" width="280" /> |
| **Default Oval:** Smooth, continuous lap navigation with zero control chatter and tight centerline lock ($d \approx 0$). | **S-Curve:** Successfully navigates reverse curves and continuous chicanes with smooth steering adjustments. |

## Roadmap

- [x] **Phase 1: Repository Architecture & Contracts Scaffolding**
  * Establish DTOs, interfaces, and directory tree configurations.
- [x] **Phase 2: Simulation Physics & Raycasting**
  * Implement the Kinematic 2D bicycle dynamics model.
  * Implement coordinate centerline boundary check queries.
  * Implement numerical raymarching for simulated Lidar range sweeps.
- [x] **Phase 3: Reinforcement Learning & Continuous Policy Control**
  * Implement Gymnasium environment wrapper (`RCCarEnv`) and Frenet reward calculator (`RewardCalculator`).
  * Implement Advantage Actor-Critic (`A2CAgent`) baseline with online 1-step TD learning.
  * Implement `RolloutBuffer` with Generalized Advantage Estimation (`GAE-λ`) and mini-batch SGD.
  * Implement Proximal Policy Optimization (`PPOAgent`) with clipped surrogate loss and warm-start checkpointing.
  * Implement Soft Actor-Critic (`SACAgent`) off-policy maximum-entropy actor-critic setup with twin Q-networks.
  * Implement automated simulation GIF recorder (`scripts/record_models.py`) and model progression tracking.
- [ ] **Phase 4: Frame Stacking & Multi-Track Curriculum Training (Next)**
  * Implement 4-frame observation stacking to provide temporal velocity/acceleration state.
  * Train policies across randomized track layouts (`default_oval`, `s_curve`, `figure_eight`).
- [ ] **Phase 5: Hardware-in-the-Loop & Physical Deployment**
  * Deploy trained PPO policy onto physical servo-driven scale RC car chassis.

## Features

* **Hardware Abstraction Layer**: Concrete classes for simulated physics backends and physical actuators bind to a common vehicle interface contract, enabling hardware-agnostic control logic.
* **Frenet Frame Transformation**: The track environment computes longitudinal displacement and lateral offsets to translate Cartesian coordinates into track-relative coordinates, simplifying state observations for Reinforcement Learning.
* **Mock-in-the-Loop Simulation**: Includes a pure-Python 2D Kinematic bicycle physics engine and numerical raymarching Lidar simulator for rapid offline unit testing and verification.
* **Decoupled Observability**: Observability tools like telemetry logging (CSV) and matplotlib dashboards run as subscriber observers to the main control loops, preventing telemetry logging code from interfering with active steering.

## Policy Progression

This section tracks the evolutionary progression of our Reinforcement Learning policies, showcasing empirical driving behaviors across reward function iterations.

| Model Checkpoint | Demonstration | Behavior & Characteristics |
| :--- | :--- | :--- |
| **`a2c_policy.pth`**<br>*(A2C Baseline)* | <img src="docs/media/a2c_policy_baseline.gif" width="220" /> | **Wall-Hugging Behavior:** Navigates track without hard collisions but settles into a sub-optimal equilibrium hugging outer track boundaries rather than maintaining lane center ($d \approx 0$). |
| **`a2c_policy_redesign_1.pth`**<br>*(A2C Redesign 1)* | <img src="docs/media/a2c_policy_redesign_1.gif" width="220" /> | **Tight Lane Centering & Steering Jerk:** Introduced Gaussian centering ($\exp(-(d/\sigma)^2)$). Achieves tight centerline tracking ($d \approx 0$), but exhibits high-frequency steering chatter due to un-penalized action rates in A2C. |
| **`PPO_policy_exploites...pth`**<br>*(PPO Attempt 1)* | <img src="docs/media/ppo_policy_rotation_exploit.gif" width="220" /> | **Rotation Exploit:** Contained an additive `+ 0.5 * centering_factor` reward term independent of speed. Discovered a reward-farming exploit spinning continuously in place near $d \approx 0$ to collect positive returns without driving forward. |
| **`PPO_policy_update_reward.pth`**<br>*(PPO Attempt 2)* | <img src="docs/media/ppo_policy_update_reward.gif" width="220" /> | **Centerline Lock & Smoothness Gain:** Removed the additive `+ 0.5 * centering_factor` term, forcing `centering_factor` to act strictly multiplicatively with forward velocity progress. Completely eliminated rotation exploits with major smoothness gains over A2C. |
| **`PPO_policy_update_reward_1.pth`**<br>*(PPO Attempt 3)* | <img src="docs/media/ppo_policy_update_reward_1.gif" width="220" /> | **Warm-Started Lap Completion:** Built directly on top of Attempt 2 by initializing from its partially trained weights and continuing optimization. Delivers smooth, highly polished lap navigation. |

For detailed analysis, reward engineering observations, and training stability dynamics, see [models/README.md](models/README.md).

## Repository Structure

The codebase is organized as follows:

```text
RC-car/
├── docs/
│   └── architecture.md         # Architecture diagrams and telemetry dataflow descriptions
├── models/                     # Trained RL policy checkpoints (.pth) and progression documentation
├── scripts/                    # Runnable user-facing entry points
│   ├── simulate_car.py         # Scripted simulation run that logs telemetry to CSV
│   ├── keyboard_drive.py       # Terminal manual key-mapping teleoperation drive cockpit
│   ├── drive_visualize.py      # Real-time Matplotlib visualization dashboard
│   └── autonomous_drive.py     # Main loop executing computer vision or policy self-driving
│
├── src/                        # Core library source code
│   ├── common/                 # Contracts, DTOs, and abstract base classes (no dependencies)
│   │   ├── backend.py          # Abstract CarBackend interface contract
│   │   ├── sensor.py           # Abstract LidarDevice sensor contract
│   │   └── types.py            # Common data structures (CarTelemetry, CarCommand, FrenetState)
│   │
│   ├── drive/                  # Guidance, Navigation, and Control
│   │   ├── controller.py       # High-level trajectory planning commands
│   │   ├── rl_controller.py    # Policy inference command controller
│   │   ├── sim_backend.py      # Kinematic 2D bicycle dynamics simulator
│   │   └── collision_avoid.py  # Proximity emergency braking safety controller
│   │
│   ├── environment/            # Map configurations and boundary definitions
│   │   ├── track.py            # Centerline waypoints and boundary limits
│   │   └── obstacles.py        # Database of static and dynamic obstacles
│   │
│   ├── perception/             # Sensor processing and computer vision
│   │   ├── lane_detector.py    # OpenCV lane detection offset estimators
│   │   ├── lidar_scan.py       # Noise filters and clustering for scanner returns
│   │   ├── lidar_sim.py        # Raymarching simulator calculating virtual ranges
│   │   └── lidar_driver.py     # Serial port drivers for physical laser sweeps
│   │
│   ├── rl/                     # Reinforcement Learning setups
│   │   ├── env.py              # Gymnasium environment wrapping car and sensors
│   │   ├── agent.py            # Neural network policy architectures
│   │   └── rewards.py          # Reward shaping calculators
│   │
│   ├── state/                  # Aggregators and health estimators
│   │   └── car_state.py        # Statistics tracker (odometry, battery check)
│   │
│   └── tools/                  # Observability and teleoperation
│       ├── telemetry_logger.py # CSV telemetry log writer observer
│       └── keyboard_teleop.py  # keystroke-to-actuator command mapper
│
└── tests/                      # Fast, offline test suite
    ├── test_mvp_car.py         # Verifies basic kinematics and safety in sim
    └── test_perception.py      # Verifies lane line parsing math
```

## Setup and Getting Started

### 1. Initialize Virtual Environment
Set up a local Python virtual environment and install project dependencies:
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Scaffolding Installation
Run the unit test suite to verify that packages import correctly and no syntax errors are present:
```bash
python3 -m pytest tests/
```

### 3. Run Autonomous Driving Inference (`autonomous_drive.py`)
Run autonomous policy inference with real-time Matplotlib visualization using [scripts/autonomous_drive.py](scripts/autonomous_drive.py):

```bash
# Run PPO Attempt 3 (top-performing model) on default_oval track
python scripts/autonomous_drive.py --algo PPO --weights models/PPO_policy_update_reward_1.pth --track default_oval

# Run PPO Attempt 3 zero-shot on s_curve track
python scripts/autonomous_drive.py --algo PPO --weights models/PPO_policy_update_reward_1.pth --track s_curve
```

#### Command-Line Arguments:
* `--algo`: Policy algorithm (`PPO`, `A2C`, `SAC`, or `RANDOM`). Default: `A2C`.
* `--weights`: Path to trained PyTorch weights checkpoint file (`.pth`).
* `--track`: Target track layout (`default_oval`, `s_curve`, `figure_eight`). Default: `default_oval`.
* `--steps`: Maximum simulation steps to execute (default: `1000`).

## License

TBD
