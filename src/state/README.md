# state/ Package

## Purpose
This package aggregates, processes, and maintains historical data logs and runtime state computations (e.g. mileage tracked, average speed, battery remaining, warning/error status).

## Level of Abstraction
* **Level**: State representation & health metrics estimator
* **Role**: Provides a clean interface to query derived metrics from the stream of incoming telemetry snapshots.
