# System Architecture & Telemetry Dataflow

This document explains the software architecture, modular component design, and dataflow pipelines for the Autonomous RC-Car platform.

*Note: Certain scaffolded modules (e.g., physical hardware drivers, alternative agents) currently define interface contracts and method signatures, which will be fully implemented gradually as physical hardware deployment progresses.*

---

## Component Block Diagram

The software layout enforces a strict separation of concerns, decoupling physical and simulated dynamics backends, sensor perception, guidance and control (GNC), Reinforcement Learning policies, and observability tools:

```text
                                 +------------------------+
                                 |   src/environment/     | (Track Geometry & Obstacles)
                                 +------------------------+
                                              ^
                                              | Queries boundaries & Frenet projection
                                              v
+------------------+             +------------------------+             +-----------------------+
|  src/perception/ | ----------> |   src/drive/           | <---------- |     src/common/       |
|  (Lane Detect,   |             |   (CarController,      |             |     (CarBackend &     |
|   Lidar Simulator|             |    RLCarController,    |             |      LidarDevice      |
|   Lidar Driver)  |             |    SimBackend)         |             |      Contracts, DTOs) |
+------------------+             +------------------------+             +-----------------------+
         ^                                    ^                                     ^
         | Raycasts                           | Commands                            |
         v                                    v                                     | Imports
+------------------+                          |                                     | DTOs
|   src/rl/        | <------------------------+                                     |
|   (RCCarEnv,     |                                                                |
|    Agents & PPO, | ---------------------------------------------------------------+
|    Actor/Critic, |
|    RolloutBuffer)|
+------------------+
         |
         | Emits Telemetry Ticks
         v
+------------------+
|   src/state/     | (CarState Aggregators)
+------------------+
         |
         | Broadcasts updates
         v
+------------------+
|   src/tools/     | (TrackVisualizer,
|  & scripts/      |  TelemetryLogger,
|                  |  Autonomous Drive UI)
+------------------+
```

---

## 1. Abstraction Levels

* **`src/common/` (Contracts & Interfaces)**: Defines core data transfer objects (`CarTelemetry`, `CarCommand`, `FrenetState`), vehicle actuator contracts (`CarBackend`), and sensor scanner contracts (`LidarDevice`). Has zero external dependencies.
* **`src/drive/` (Guidance, Navigation & Control)**: Executes continuous motion control (`CarController`), 2D kinematic bicycle physics (`SimulatedCar`), proximity safety braking (`CollisionAvoidance`), and policy-driven inference (`RLCarController`).
* **`src/environment/` (World Geometry & Maps)**: Models track centerline waypoints (`Track`), Frenet frame coordinate transformations ($s, d, \theta_{\text{err}}$), wall boundary lines, and static/dynamic obstacle collections (`ObstacleMap`).
* **`src/perception/` (Sensors & Vision)**: Evaluates camera frames for lane boundaries (`LaneDetector`), processes point-cloud sweeps (`LidarScan`), and implements virtual raymarching (`LidarSimulator`) and serial physical hardware (`PhysicalLidarDriver`) sensors.
* **`src/rl/` (Reinforcement Learning Framework)**:
  * `agents/`: Modular RL agents (`PPOAgent`, `A2CAgent`, `SACAgent`, `RandomAgent`).
  * `networks/`: PyTorch neural architectures (`ActorNetwork`, `CriticNetwork`, `SharedMLP`).
  * `base_agent.py`: Abstract `BaseAgent` policy interface contract.
  * `env.py`: Standard Gymnasium environment wrapper (`RCCarEnv`).
  * `rewards.py`: Frenet progress, Gaussian centering, and steering jerk reward shaping calculators (`RewardCalculator`).
  * `rollout_buffer.py`: Fixed-capacity GAE rollout trajectory buffer (`RolloutBuffer`).
* **`src/state/` (Telemetry Processing)**: Aggregates real-time values to track mileage, speed, heading, and vehicle health metrics (`CarState`).
* **`src/tools/` (Observability & Teleoperation)**: Provides real-time Matplotlib visualization dashboards (`TrackVisualizer`), CSV log writers (`TelemetryLogger`), and teleoperation mappers (`KeyboardTeleop`).
* **`scripts/` (User-Facing Entry Points)**: Command-line entry points for autonomous inference (`autonomous_drive.py`), training (`train_rl.py`), visualizer demonstration (`drive_visualize.py`), GIF recording (`record_models.py`), and simulation telemetry logging (`simulate_car.py`).

---

## 2. Telemetry & RL Inference Flow (`autonomous_drive.py`)

When running autonomous driving policy inference:

1. **Sensor Measurement**: `RLCarController` queries `LidarDevice` (either `LidarSimulator` or `PhysicalLidarDriver`) via `read_scan()`.
2. **State & Frenet Projection**: `SimulatedCar` telemetry $(x, y, \theta)$ is projected onto `Track` geometry to produce track-relative Frenet offset $d$ and heading error $\theta_{\text{err}}$.
3. **Observation Vector Construction**: Metrics are normalized to form the 6D observation vector:
   $$S_t = [v_{\text{norm}}, d_{\text{norm}}, \theta_{\text{err}}, \text{lidar}_1, \text{lidar}_2, \text{lidar}_3]$$
4. **Policy Forward Pass**: The observation vector is passed to `PPOAgent.select_action(obs)`, executing `ActorNetwork.get_action()` to compute continuous steering $[-1, 1]$ and throttle $[0, 1]$ commands.
5. **Dynamics Integration**: Commands are sent to `CarBackend` (`SimulatedCar`), updating bicycle kinematic equations over time step $\Delta t = 0.05\text{s}$.
6. **Observability Update**: Telemetry and Lidar scans are passed to `TrackVisualizer`, updating the live plot canvas and rendering trajectory overlays.

---

## 3. Reinforcement Learning Training Pipeline (`train_rl.py`)

### 3.1 On-Policy PPO Pipeline (`PPOAgent`)

During PPO policy training:

```text
+-----------------------+     Step Action     +-----------------------+
|  PPOAgent / Actor     | ------------------> |       RCCarEnv        |
|  (Gaussian Sampling)  |                     |  (Kinematics & Track) |
+-----------------------+                     +-----------------------+
            ^                                             |
            |                                             | Returns (obs, reward, done)
            | Batch SGD Optimization                      v
+---------------------------------------------------------------------+
|                          RolloutBuffer                              |
|  - Stores trajectories (s_t, a_t, r_t, v_t, log_prob_t)             |
|  - Calculates GAE advantages: A_t = \sum (\gamma \lambda)^k \delta  |
|  - Computes target returns: G_t = A_t + V(s_t)                       |
+---------------------------------------------------------------------+
```

1. **Trajectory Rollout Collection**: `train_rl.py` steps `RCCarEnv` for $N=2048$ steps, storing transitions into `RolloutBuffer`.
2. **Advantage Estimation**: `RolloutBuffer` computes Generalized Advantage Estimation ($\text{GAE}-\lambda$) backward-in-time:
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
3. **Mini-Batch PPO SGD Update**: `PPOAgent._ppo_update()` samples mini-batches over $K=10$ epochs, evaluating clipped surrogate policy loss:
   $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
4. **Checkpoint Preservation**: Models are serialized to PyTorch weight checkpoints (`models/ppo/*.pth`).

---

### 3.2 Off-Policy Soft Actor-Critic Pipeline (`SACAgent`)

During SAC policy training:

```text
+-----------------------+     Step Action     +-----------------------+
|  SACActor (Policy)    | ------------------> |       RCCarEnv        |
|  a = tanh(u)          |                     |  (Kinematics & Track) |
+-----------------------+                     +-----------------------+
            ^                                             |
            | Reparameterized Backprop                    | Stores (s, a, r, s', term)
            | \nabla_\theta a_\theta(s)                   v
+-----------------------+                     +-----------------------+
| Twin Critics (Q1, Q2) | <------------------ |     ReplayBuffer      |
| & Soft Targets        |   Sample Mini-Batch |  (Circular Ring, 50k) |
+-----------------------+   (B = 64)          +-----------------------+
```

1. **Warmup Exploration & Collection**: For the first 5,000 steps (`start_step`), actions are drawn uniformly from `env.action_space.sample()`. Transitions $(s, a, r, s', \text{terminated})$ are saved to `ReplayBuffer`.
2. **Sample Mini-Batch**: A random batch ($B=64$) is sampled at every step.
3. **Soft Bellman Critic Update**: Evaluates minimum target Q with entropy correction:
   $$y = r + \gamma (1 - d) \left( \min(Q_1^{\text{targ}}(s', a'), Q_2^{\text{targ}}(s', a')) - \alpha \log \pi(a'|s') \right)$$
   Minimizes $\mathcal{L}_Q = \mathbb{E}[(Q_1(s, a) - y)^2] + \mathbb{E}[(Q_2(s, a) - y)^2]$.
4. **Actor Policy Update**: Maximizes expected Q-value with entropy regularizer via reparameterization trick:
   $$\mathcal{L}_\pi = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ \min(Q_1(s, a), Q_2(s, a)) - \alpha \log \pi(a|s) \right]$$
5. **Polyak Target Tracking**: Continuously updates target critic parameters at every step:
   $$\theta_{\text{target}} \leftarrow 0.995 \cdot \theta_{\text{target}} + 0.005 \cdot \theta_{\text{current}}$$
6. **Checkpoint Preservation**: Intermediate and final models are saved to `models/sac/version4/*.pth`.
