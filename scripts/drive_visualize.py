# """Real-time visualizer dashboard: plotting trajectory, vehicle state, and Lidar scans.
#
# This script boots the 2D kinematics simulator backend and track environment,
# then renders a real-time Matplotlib interactive visualizer displaying the track boundaries,
# historical vehicle trajectory path, car heading, and live Lidar ray sweeps.
# """

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
from src.drive.sim_backend import SimulatedCar
from src.environment.track import Track
from src.environment.obstacles import ObstacleMap
from src.perception.lidar_sim import LidarSimulator
from src.tools.visualizer import TrackVisualizer
from src.common.types import CarCommand

def main(track_name = "default_oval") -> None:
    """Run interactive 2D simulation visualizer demonstration loop."""
    print("Initializing 2D Simulation Visualizer Dashboard...")
    
    # 1. Instantiate Track, Physics Simulator, and Lidar Raycaster
    track = Track(track_name=track_name, track_width=1.6)
    obs_map = ObstacleMap()
    car = SimulatedCar(wheelbase=0.25, max_speed=5.0)
    car.connect()
    
    lidar_sim = LidarSimulator(obstacle_map=obs_map, backend=car, track=track, num_rays=5, max_range_m=5.0)
    
    # 2. Instantiate TrackVisualizer
    visualizer = TrackVisualizer(track=track, title="RC Car 2D Kinematic Simulator & Lidar Raycaster")
    
    print("Running simulation drive loop... (Close visualization window to exit)")
    
    dt = 0.05
    steps = 200
    
    for step in range(steps):
        # Apply sample control inputs: accelerate forward and steer gradually around track curves
        steering_input = 0.3 * math.sin(step * 0.05)
        car.send_command(CarCommand(throttle=0.6, steering=steering_input, brake=0.0))
        
        # Step physics dynamics
        car.update(dt=dt)
        
        # Query telemetry, Lidar rays, and Frenet state
        telemetry = car.telemetry()
        scan = lidar_sim.read_scan()
        frenet = track.cartesian_to_frenet(telemetry.x, telemetry.y, telemetry.heading_deg)
        
        # Update visualizer frame
        visualizer.update(telemetry=telemetry, scan=scan, frenet=frenet)
        
        # Pause slightly for real-time visualization playback
        time.sleep(0.02)
        
        # Check boundary collision status
        if not track.is_within_boundaries(telemetry.x, telemetry.y):
            print(f"Step {step}: Car went off-track! Coordinates: ({telemetry.x:.2f}, {telemetry.y:.2f})")
            break

    print("Simulation drive visualization complete. (Close window to exit)")
    visualizer.keep_open()

if __name__ == "__main__":
    # figure_eight or s_curve or default_oval
    main(track_name="s_curve")
