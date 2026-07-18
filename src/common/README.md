# common/ Package

## Purpose
This package defines the core types, enums, data structures, and abstract base classes for the RC Car software. It contains no execution logic or algorithmic code.

## Level of Abstraction
* **Level**: Hardware/Simulation Interface Contract

## Relations to Other Components
Everything else plugs into this. Start by importing the abstract interfaces (`CarBackend`) and telemetry dataclasses (`CarTelemetry`). Concrete simulators, control loops, and perception modules implement or consume these data types to guarantee that your guidance controller runs identically on either a math simulation or a real servo-driven chassis.
