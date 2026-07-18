# tools/ Package

## Purpose
This package provides observability, telemetry logging (to CSV/JSON), and human interface devices teleoperation command-mapping logic.

## Level of Abstraction
* **Level**: Auxiliary tools & Observability utilities

## Relations to Other Components
These are the peripheral observers and manual interfaces. The `telemetry_logger.py` module registers as a callback subscriber to the controller in `src/drive/` to log telemetry ticks without altering control flows, while `keyboard_teleop.py` intercepts console key strokes to pass manual override targets directly into the active vehicle backend.
