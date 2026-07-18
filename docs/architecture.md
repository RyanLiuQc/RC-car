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
|  (Lane Detect,   |      |   (CarController,      |      |     (Abstract Backend |
|   Lidar Process) |      |    SimBackend)         |      |      & Types Contract)|
+------------------+      +------------------------+      +-----------------------+
                                       |
                                       | Emits Telemetry Tick
                                       v
                          +------------------------+
                          |   src/state/           | (CarState Aggregators)
                          +------------------------+
                                       |
                                       | Notifies
                                       v
                          +------------------------+
                          |   src/tools/           | (Telemetry Observers,
                          |   & drive_visualize    |  e.g., CSV Logger, UI)
                          +------------------------+
```

---

## 1. Abstraction Levels

* **`src/common/` (Contract & Interfaces)**: Defines the system types (`CarTelemetry`) and the backend contract (`CarBackend`). It has no external dependencies.
* **`src/drive/` (Kinematics & Control)**: Executes motion control (`CarController`), kinematic simulations (`SimulatedCar`), and safety constraints (`CollisionAvoidance`).
* **`src/environment/` (Virtual Map Map/Obstacles)**: Models the static boundaries, lane markers, and obstacles in the simulated space.
* **`src/perception/` (Sensors & CV)**: Evaluates camera frames for lanes and Lidar signals for obstacle ranges.
* **`src/state/` (Telemetry Processing)**: Aggregates real-time values to calculate battery status, mileage, and errors.
* **`src/tools/` (Observability & I/O)**: Logs data to CSV or maps external signals (keyboard inputs) to vehicle controls.

---

## 2. Telemetry Flow

The driving sequence runs on an event-driven tick loop:
1. **Control Command**: The `CarController` computes throttle and steering controls.
2. **Backend Update**: Commands are sent to the `CarBackend` (either the `SimulatedCar` or a real hardware wrapper).
3. **Sensor Scan**: `lidar_sim` queries the `environment/obstacles` to evaluate virtual distances.
4. **State Emitted**: The backend builds a `CarTelemetry` snapshot.
5. **State Aggregation**: `CarState` intercepts telemetry to check battery and compute running metrics.
6. **Listener Notification**: Any registered listener (e.g. `TelemetryLogger`, Matplotlib dashboard) receives the telemetry update.
