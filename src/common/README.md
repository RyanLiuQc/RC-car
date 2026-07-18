# common/ Package

## Purpose
This package defines the core types, enums, data structures, and abstract base classes for the RC Car software. It contains no execution logic or algorithmic code.

## Level of Abstraction
* **Level**: Hardware/Simulation Interface Contract

## Relations to Other Components
Everything else plugs into this. Start by importing the abstract interfaces (`CarBackend`, `LidarDevice`) and telemetry dataclasses (`CarTelemetry`). Concrete simulators, control loops, sensor drivers, and perception modules implement or consume these data types to guarantee that your guidance controller and perception loops run identically on either math simulations or real hardware interfaces.
