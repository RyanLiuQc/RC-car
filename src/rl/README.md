# rl/ Package

## Purpose
This package provides the reinforcement learning infrastructure. It defines the training environment wrapper, neural network policy models, and reward shaping functions needed to train the autonomous driving agent.

## Level of Abstraction
* **Level**: Machine Learning Environment & Training Setup
* **Role/Relations**: 
  This is where the reinforcement learning environment, networks, and reward functions are defined. It takes the kinematic physical car backend from `src/drive/` and the simulated obstacle scanners from `src/perception/`, wrapping them inside a standard Gymnasium interface for training policies. The resulting policy models are then consumed by the controllers in `src/drive/` to execute autonomous drives.

## Gymnasium Environment (`RCCarEnv`)

The custom Gymnasium environment wraps the simulation physics backend, track geometry, and raymarching Lidar sensor into an OpenAI Gymnasium compliant interface (`gym.Env`).

### Action Space (`gym.spaces.Box`)
The action space is a 1D continuous array of shape `(2,)`:

| Index | Action Variable | Range | Description |
| :--- | :--- | :--- | :--- |
| `0` | `throttle` | `[-1.0, 1.0]` | Normalized acceleration / braking request (`-1.0` full brake, `1.0` full throttle) |
| `1` | `steering` | `[-1.0, 1.0]` | Normalized front-wheel steering request (`-1.0` full left, `1.0` full right) |

### Observation Space (`gym.spaces.Box`)
The observation space is a 1D continuous array of shape `(6,)`:

| Index | Observation Variable | Physical Unit | Min Bound | Max Bound | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `speed_mps` | meters / second (m/s) | `0.0` | `5.0` | Vehicle forward linear velocity |
| `1` | `d` | meters (m) | `-2.0` | `2.0` | Frenet lateral offset from track centerline (positive left, negative right) |
| `2` | `heading_error_deg` | degrees (deg) | `-180.0` | `180.0` | Yaw angle error relative to track segment tangent |
| `3` | `lidar_right` | meters (m) | `0.0` | `5.0` | Right Lidar ray distance (-30 deg sweep) |
| `4` | `lidar_front` | meters (m) | `0.0` | `5.0` | Center Lidar ray distance (0 deg sweep) |
| `5` | `lidar_left` | meters (m) | `0.0` | `5.0` | Left Lidar ray distance (+30 deg sweep) |

## Reward Function (`RewardCalculator`)

The reward calculator produces a scalar step reward based on progress, tracking precision, and collision avoidance:

* **Progress / Speed Reward**: Rewards maintaining target cruise speed (1.5 m/s).
* **Lateral Displacement Penalty**: Penalizes distance d from the track centerline.
* **Heading Error Penalty**: Penalizes misalignment delta_theta with the track direction.
* **Terminal Crash Penalty**: Applies a large negative penalty (-100.0) and terminates the episode if the vehicle crosses track boundaries or hits obstacles.
