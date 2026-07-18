# perception/ Package

## Purpose
This package houses lane-line computer vision code, Lidar scan processing, and virtual Lidar sensor simulations for offline test runs.

## Level of Abstraction
* **Level**: Low-level sensor drivers & computer vision estimation

## Relations to Other Components
This is the observation generator of the vehicle. It translates camera frames or lidar scans into lane errors and distances, which are consumed by the collision check and navigation logic in `src/drive/`. Its simulation submodule `lidar_sim.py` queries `src/environment/` to calculate mock obstacle range measurements based on the vehicle's simulated coordinates.
