# Autonomous RC-Car Project

Welcome to the **Autonomous RC-Car** software platform. This project implements a modular, clean, and highly decoupled architecture designed to drive an autonomous 1/10th scale (or simulated) RC car.

It features a 2D Kinematic simulation model, support for manual teleoperation (keyboard), computer vision lane-following, ultrasonic/Lidar collision avoidance, and state telemetry logging.

---

## Repository Directory Layout

```text
RC-car/
├── docs/
│   └── architecture.md         # Document outlining component design, interfaces, and flow
├── scripts/                    # Runnable user-facing entry points
│   ├── simulate_car.py         # Scripted demo drive, logs telemetry CSV
│   ├── keyboard_drive.py       # Drive cockpit using arrow keys or terminal commands
│   ├── drive_visualize.py      # Matplotlib visualizer showing path, speed, battery
│   └── autonomous_drive.py     # Autonomous driving (lane keeping/waypoint navigation)
│
├── src/                        # Library source code
│   ├── common/                 # Contracts and shared data classes (no external libraries)
│   ├── drive/                  # Guidance, Navigation, and Control (contains RL controllers)
│   ├── environment/            # Map and obstacle database
│   ├── perception/             # Sensors and computer vision
│   ├── rl/                     # Gymnasium environments, network models, and reward shaping
│   ├── state/                  # Aggregators and estimators
│   └── tools/                  # Observability and teleoperation
│
├── tests/                      # Testing suite
│   ├── test_mvp_car.py         # Sim-based driving tests and collision avoidance validation
│   └── test_perception.py      # Tests for lane tracking algorithms on synthetic frames
│
├── requirements.txt            # Project dependencies (numpy, opencv-python, matplotlib, pytest)
├── README.md                   # Welcome page, build guides, and run commands
└── CONTRIBUTING.md             # Contribution, formatting, and PR submission guidelines
```

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Tests**:
   ```bash
   python3 -m pytest tests/
   ```
