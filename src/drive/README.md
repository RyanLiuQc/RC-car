# drive/ Package

## Purpose
This package implements Guidance, Navigation, and Control (GNC) algorithms. It is responsible for driving control loops, simulated kinematics, and basic safety guardrails like emergency stopping.

## Level of Abstraction
* **Level**: High-level control algorithms & simulation physics
* **Role**: Translates high-level mission requests (e.g. drive a distance, follow a path) or safety events into low-level throttle and steering inputs using the abstract backend contract.
