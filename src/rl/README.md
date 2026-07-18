# rl/ Package

## Purpose
This package provides the reinforcement learning infrastructure. It defines the training environment wrapper, neural network policy models, and reward shaping functions needed to train the autonomous driving agent.

## Level of Abstraction
* **Level**: Machine Learning Environment & Training Setup
* **Role/Relations**: 
  This is where the reinforcement learning environment, networks, and reward functions are defined. It takes the kinematic physical car backend from `src/drive/` and the simulated obstacle scanners from `src/perception/`, wrapping them inside a standard Gymnasium interface for training policies. The resulting policy models are then consumed by the controllers in `src/drive/` to execute autonomous drives.
