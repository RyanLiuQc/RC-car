# perception/ Package

## Purpose
This package houses lane-line computer vision code, Lidar scan processing, physical rotating laser drivers, and virtual Lidar sensor simulations for offline test runs.

## Level of Abstraction
* **Level**: Low-level sensor drivers & computer vision estimation

## Relations to Other Components
This is the observation generator of the vehicle. It translates camera frames or lidar scans into lane errors and distances, which are consumed by the collision check and navigation logic in `src/drive/`. Sensor drivers (like `lidar_driver.py` for physical hardware or `lidar_sim.py` for simulations) implement the abstract `LidarDevice` contract, allowing controllers to acquire scans polymorphically. The `lidar_sim.py` module queries the active backend telemetry coordinates and the `src/environment/` track configuration to calculate mock obstacle range measurements.
