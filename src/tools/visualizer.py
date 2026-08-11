# """Real-time Matplotlib visualizer dashboard: plotting car, track boundaries, and Lidar rays.
#
# This tool renders real-time 2D graphics showing the track centerline, wall boundaries,
# historical vehicle trajectory, car orientation, and active Lidar scanner rays.
# """

import math
import matplotlib.pyplot as plt
from typing import List, Optional
from src.environment.track import Track
from src.common.types import CarTelemetry, LidarScan, FrenetState

class TrackVisualizer:
    def __init__(self, track: Optional[Track] = None, title: str = "RC Car Real-Time Visualizer") -> None:
        self.track = track
        self.x_history: List[float] = []
        self.y_history: List[float] = []

        # Setup interactive Matplotlib figure
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle(title, fontsize=12, fontweight='bold')
        
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_aspect("equal", adjustable="datalim")

        # Plot static track centerline & boundary walls if track is provided
        if self.track:
            self._draw_track_layout()

        # Dynamic plot elements
        (self.traj_line,) = self.ax.plot([], [], "b-", linewidth=1.5, label="Trajectory", alpha=0.7)
        (self.car_dot,) = self.ax.plot([], [], "ro", markersize=8, label="Car")
        self.heading_arrow = None
        self.lidar_lines: List[plt.Line2D] = []
        
        # Info overlay text
        self.info_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        
        self.ax.legend(loc="upper right")

    def _draw_track_layout(self) -> None:
        """Render track centerline and left/right boundary walls."""
        if not self.track:
            return

        # 1. Draw Centerline Waypoints
        wx = [w[0] for w in self.track.waypoints] + [self.track.waypoints[0][0]]
        wy = [w[1] for w in self.track.waypoints] + [self.track.waypoints[0][1]]
        self.ax.plot(wx, wy, "k--", linewidth=1.0, label="Centerline", alpha=0.5)

        # 2. Draw Boundary Walls
        for seg in self.track.boundaries:
            x1, y1, x2, y2 = seg
            self.ax.plot([x1, x2], [y1, y2], "k-", linewidth=1.5, alpha=0.8)

    def update(
        self,
        telemetry: CarTelemetry,
        scan: Optional[LidarScan] = None,
        frenet: Optional[FrenetState] = None
    ) -> None:
        """
        Update the visualizer frame with vehicle position, heading, and Lidar rays.
        """
        # Append trajectory history
        self.x_history.append(telemetry.x)
        self.y_history.append(telemetry.y)
        self.traj_line.set_data(self.x_history, self.y_history)

        # Update Car position marker
        self.car_dot.set_data([telemetry.x], [telemetry.y])

        # Remove previous heading arrow
        if self.heading_arrow:
            self.heading_arrow.remove()

        # Draw vehicle heading arrow (length = 0.4m)
        heading_rad = math.radians(telemetry.heading_deg)
        arrow_dx = 0.4 * math.cos(heading_rad)
        arrow_dy = 0.4 * math.sin(heading_rad)
        self.heading_arrow = self.ax.arrow(
            telemetry.x, telemetry.y, arrow_dx, arrow_dy,
            head_width=0.15, head_length=0.15, fc="red", ec="red", zorder=5
        )

        # Remove previous Lidar rays
        for line in self.lidar_lines:
            line.remove()
        self.lidar_lines.clear()

        # Draw active Lidar rays
        if scan and len(scan.angles_deg) == len(scan.ranges_m):
            for angle_deg, dist_m in zip(scan.angles_deg, scan.ranges_m):
                ray_angle_rad = math.radians(telemetry.heading_deg + angle_deg)
                hit_x = telemetry.x + dist_m * math.cos(ray_angle_rad)
                hit_y = telemetry.y + dist_m * math.sin(ray_angle_rad)
                
                # Red line for ray hit
                (line,) = self.ax.plot(
                    [telemetry.x, hit_x], [telemetry.y, hit_y],
                    "g-", linewidth=1.0, alpha=0.6
                )
                self.lidar_lines.append(line)

        # Update overlay text
        info = f"Speed: {telemetry.speed_mps:.2f} m/s\nHeading: {telemetry.heading_deg:.1f}°\nBattery: {telemetry.battery_pct:.1f}%"
        if frenet:
            info += f"\nFrenet s: {frenet.s:.2f}m | d: {frenet.d:.2f}m"
        self.info_text.set_text(info)

        # Refresh figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Close interactive plot window."""
        plt.ioff()
        plt.close(self.fig)

    def keep_open(self) -> None:
        """Keep plot window open blocking until manually closed by user."""
        plt.ioff()
        plt.show(block=True)

