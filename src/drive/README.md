# drive/ Package

## Purpose
This package implements Guidance, Navigation, and Control (GNC) algorithms. It is responsible for driving control loops, simulated kinematics, and basic safety guardrails like emergency stopping.

## Level of Abstraction
* **Level**: High-level control algorithms & simulation physics

## Relations to Other Components
This is the active control layer of the vehicle. It consumes the abstract interface contract from `src/common/` to execute guidance steps, implements the kinematic physics model simulating the car's state, and receives processed obstacle metrics from `src/perception/` to run collision checks and override actions. It acts as the execution block driven by the runner scripts.
