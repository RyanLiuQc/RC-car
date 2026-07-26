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

| Index | Observation Variable | Physical Unit | Raw Min / Max | Normalized Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `speed_mps` | meters / second (m/s) | `0.0` to `5.0` | `[0.0, 1.0]` | Scaled vehicle linear velocity (`speed / 5.0`) |
| `1` | `d` | meters (m) | `-2.0` to `2.0` | `[-1.0, 1.0]` | Scaled Frenet lateral offset from centerline (`d / 1.0`) |
| `2` | `heading_error_deg` | degrees (deg) | `-180.0` to `180.0` | `[-1.0, 1.0]` | Scaled yaw angle error (`heading_error / 180.0`) |
| `3` | `lidar_right` | meters (m) | `0.0` to `5.0` | `[0.0, 1.0]` | Scaled right Lidar ray distance (`range / 5.0`) |
| `4` | `lidar_front` | meters (m) | `0.0` to `5.0` | `[0.0, 1.0]` | Scaled center Lidar ray distance (`range / 5.0`) |
| `5` | `lidar_left` | meters (m) | `0.0` to `5.0` | `[0.0, 1.0]` | Scaled left Lidar ray distance (`range / 5.0`) |

## Reward Function (`RewardCalculator`)

The reward calculator produces a scalar step reward based on progress, tracking precision, and collision avoidance:

* **Progress / Speed Reward**: Rewards maintaining target cruise speed (1.5 m/s).
* **Lateral Displacement Penalty**: Penalizes distance d from the track centerline.
* **Heading Error Penalty**: Penalizes misalignment delta_theta with the track direction.
* **Terminal Crash Penalty**: Applies a large negative penalty (-100.0) and terminates the episode if the vehicle crosses track boundaries or hits obstacles.
