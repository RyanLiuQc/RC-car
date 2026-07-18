# environment/ Package

## Purpose
This package defines the physical and geometry properties of the virtual simulated track, boundaries, and static/dynamic obstacles.

## Level of Abstraction
* **Level**: World Environment Mapping

## Relations to Other Components
This models the physical space in which the simulation runs. The kinematic physics simulation in `src/drive/` queries the track waypoints to guide navigation, while simulated sensors in `src/perception/` query the obstacles map to compute virtual lidar distances. It has no dependencies on control or state logging, acting as a standalone virtual world layout database.
