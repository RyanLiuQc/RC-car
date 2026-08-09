# Autonomous RC-Car Control Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

## Overview

This repository contains a modular, clean, and highly decoupled software platform designed for autonomous driving development. It provides a structured foundation that enables a seamless transition between a 2D Kinematic Simulation environment and a physical, servo-driven 1/8 scale RC car chassis. 

By enforcing strict boundaries between physical dynamics, track layout geometry, sensor perception systems, and control logic, the platform supports rapid prototyping of classical controllers, computer vision pipelines, and Reinforcement Learning policies.

## Roadmap

- [x] **Phase 1: Repository Architecture & Contracts Scaffolding**
  * Establish DTOs, interfaces, and directory tree configurations.
- [x] **Phase 2: Simulation Physics & Raycasting**
  * Implement the Kinematic 2D bicycle dynamics model.
  * Implement coordinate centerline boundary check queries.
  * Implement numerical raymarching for simulated Lidar range sweeps.
- [ ] **Phase 3: Controller & Trajectory Tracking (Next)**

## Features

* **Hardware Abstraction Layer**: Concrete classes for simulated physics backends and physical actuators bind to a common vehicle interface contract, enabling hardware-agnostic control logic.
* **Frenet Frame Transformation**: The track environment computes longitudinal displacement and lateral offsets to translate Cartesian coordinates into track-relative coordinates, simplifying state observations for Reinforcement Learning.
* **Mock-in-the-Loop Simulation**: Includes a pure-Python 2D Kinematic bicycle physics engine and numerical raymarching Lidar simulator for rapid offline unit testing and verification.
* **Decoupled Observability**: Observability tools like telemetry logging (CSV) and matplotlib dashboards run as subscriber observers to the main control loops, preventing telemetry logging code from interfering with active steering.

## Policy Progression

This section tracks the evolutionary progression of our Reinforcement Learning policies, showcasing empirical driving behaviors across reward function iterations.

| Model Checkpoint | Demonstration | Behavior & Characteristics |
| :--- | :--- | :--- |
| **`a2c_policy.pth`**<br>*(Baseline Attempt)* | ![A2C Baseline Policy](docs/media/a2c_policy_baseline.gif) | **Wall-Hugging Behavior:** Navigates track without hard collisions but settles into a sub-optimal equilibrium hugging outer track boundaries rather than maintaining lane center ($d \approx 0$). |
| **`a2c_policy_redesign_1.pth`**<br>*(Reward Redesign 1)* | ![A2C Redesign 1 Policy](docs/media/a2c_policy_redesign_1.gif) | **Tight Lane Centering & Steering Jerk:** Removed linear progress reward exploitation and introduced Gaussian centering ($\exp(-(d/\sigma)^2)$). Achieves tight centerline tracking ($d \approx 0$), but exhibits high-frequency steering chatter ("jerk") due to un-penalized action rates. |

For detailed analysis, reward engineering observations, and training stability dynamics, see [models/README.md](models/README.md).

## Repository Structure

The codebase is organized as follows:

```text
RC-car/
├── docs/
│   └── architecture.md         # Architecture diagrams and telemetry dataflow descriptions
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

## License

TBD
