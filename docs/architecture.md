# System Architecture & Telemetry Dataflow

This document explains the software architecture and component design for the Autonomous RC-Car project.

## Component Block Diagram

The software layout ensures a strict separation of concerns, decoupling the hardware communications, high-level driving behaviors, environmental layout, and monitoring tools:

```
                          +------------------------+
                          |   src/environment/     | (Map & Obstacles)
                          +------------------------+
                                       ^
                                       | Queries boundaries
                                       v
+------------------+      +------------------------+      +-----------------------+
|  src/perception/ | ---> |   src/drive/           | <--- |     src/common/       |
|  (Lane Detect,   |      |   (CarController,      |      |     (CarBackend &     |
|   Lidar Process, |      |    RLCarController,    |      |      LidarDevice      |
|   Lidar Simulator|      |    SimBackend)         |      |      Contracts, DTOs) |
|   Lidar Driver)  |      +------------------------+      +-----------------------+
+------------------+                  ^                               ^
         ^                            |                               |
         | Raycasts                   | Commands                      |
         v                            v                               |
+------------------+                  |                               | Imports
|   src/rl/        | <----------------+                               | types
|   (RCCarEnv,     |                                                  |
|    Agent/Policy) | -------------------------------------------------+
+------------------+
         |
         | Emits Telemetry Tick
         v
+------------------+
|   src/state/     | (CarState Aggregators)
+------------------+
         |
         | Notifies
         v
+------------------+
|   src/tools/     | (Telemetry Observers,
|  & drive_visualize e.g., CSV Logger, UI)
+------------------+
```

---

## 1. Abstraction Levels

* **`src/common/` (Contract & Interfaces)**: Defines the system DTOs (`CarTelemetry`), backend contract (`CarBackend`), and sensor contract (`LidarDevice`). It has no external dependencies.
* **`src/drive/` (Kinematics & Control)**: Executes motion control (`CarController`), kinematic simulations (`SimulatedCar`), safety constraints (`CollisionAvoidance`), and model-based driving (`RLCarController`).
* **`src/environment/` (Virtual Map Map/Obstacles)**: Models the static boundaries, lane markers, and obstacles in the simulated space.
* **`src/perception/` (Sensors & CV)**: Evaluates camera frames for lanes, processes Lidar signals for obstacle ranges, and implements virtual (`LidarSimulator`) and hardware (`PhysicalLidarDriver`) sensors.
* **`src/rl/` (Machine Learning setup)**: Packages training environments (`RCCarEnv`), reward functions (`rewards.py`), and model structure definitions (`PolicyNetwork`).
* **`src/state/` (Telemetry Processing)**: Aggregates real-time values to calculate battery status, mileage, and errors.
* **`src/tools/` (Observability & I/O)**: Logs data to CSV or maps external signals (keyboard inputs) to vehicle controls.

---

## 2. Telemetry Flow (Autonomous RL Driving Mode)

When driving autonomously using the trained reinforcement learning agent (Option A):
1. **Sensor Measurement**: The `RLCarController` queries `LidarDevice` (either `LidarSimulator` or `PhysicalLidarDriver`) via the generic `read_scan()` method.
2. **Observation Construction**: Telemetry metrics (speed) and the sensor's `LidarScan.ranges_m` are combined to form the state observation vector.
3. **Inference**: The observation vector is passed to the `PolicyNetwork` (from `src/rl/`), which executes a forward pass to compute target `throttle` and `steering` action values.
4. **Control Command**: The calculated action values are sent to the `CarBackend` (either `SimulatedCar` or a real hardware chassis).
5. **Dynamics Update**: The backend integrates controls to advance position/speed. In simulator mode, `LidarSimulator` queries `src/environment/obstacles` to evaluate the next step's virtual ranges.
6. **Listener Notification**: Telemetry updates are broadcast to the `CarStateManager` and observers in `src/tools/` (such as `TelemetryLogger`) to update metrics and logs.
