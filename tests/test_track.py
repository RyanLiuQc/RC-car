"""Unit tests for Track spatial queries, boundary checking, and Frenet transformations."""

import math
import pytest
from src.environment.track import Track, generate_track_boundaries
from src.common.types import FrenetState


@pytest.fixture
def default_track() -> Track:
    """Fixture providing a default Track instance."""
    return Track(track_name="default_oval", track_width=1.6)


def test_get_nearest_waypoint_index(default_track: Track) -> None:
    """Test finding the index of the nearest waypoint to given coordinates."""
    # (0.0, 0.0) is Waypoint 0
    idx_0 = default_track.get_nearest_waypoint_index(0.0, 0.0)
    assert idx_0 == 0

    # (4.1, 0.1) is closest to Waypoint 2 at (4.0, 0.0)
    idx_2 = default_track.get_nearest_waypoint_index(4.1, 0.1)
    assert idx_2 == 2

    # Nearest waypoint coordinates check
    pt = default_track.get_nearest_waypoint(4.1, 0.1)
    assert pt == (4.0, 0.0)


def test_is_within_boundaries_on_centerline(default_track: Track) -> None:
    """Test that points along the centerline segments are inside boundaries."""
    # Directly on Waypoint 0
    assert default_track.is_within_boundaries(0.0, 0.0) is True

    # Midpoint between Waypoint 0 (0,0) and Waypoint 1 (2,0)
    assert default_track.is_within_boundaries(1.0, 0.0) is True

    # Point slightly off-center (0.5m off centerline, width is 1.6m -> half-width is 0.8m)
    assert default_track.is_within_boundaries(1.0, 0.5) is True


def test_is_within_boundaries_off_track(default_track: Track) -> None:
    """Test that points beyond half track-width return False."""
    # 1.0m off centerline along straight segment (half-width is 0.8m)
    assert default_track.is_within_boundaries(1.0, 1.0) is False

    # Far away point
    assert default_track.is_within_boundaries(100.0, 100.0) is False


def test_cartesian_to_frenet_straight_segment(default_track: Track) -> None:
    """Test Cartesian to Frenet conversion along a straight horizontal segment."""
    # Point at (1.0, 0.0) with heading 0 deg (aligned with segment from (0,0) to (2,0))
    frenet = default_track.cartesian_to_frenet(x=1.0, y=0.0, heading_deg=0.0)

    assert isinstance(frenet, FrenetState)
    # Distance s along track up to (1,0) should be approx 1.0m
    assert frenet.s == pytest.approx(1.0, abs=1e-2)
    # Lateral offset d should be 0.0m on centerline
    assert frenet.d == pytest.approx(0.0, abs=1e-2)
    # Heading error should be 0.0 deg
    assert frenet.heading_error_deg == pytest.approx(0.0, abs=1e-2)


def test_cartesian_to_frenet_lateral_offset(default_track: Track) -> None:
    """Test signed lateral offset d (positive for left, negative for right)."""
    # Point at (1.0, 0.4) -> 0.4m to the left of segment (0,0)->(2,0)
    frenet_left = default_track.cartesian_to_frenet(x=1.0, y=0.4, heading_deg=0.0)
    assert frenet_left.d > 0.0
    assert abs(frenet_left.d) == pytest.approx(0.4, abs=1e-2)

    # Point at (1.0, -0.4) -> 0.4m to the right
    frenet_right = default_track.cartesian_to_frenet(x=1.0, y=-0.4, heading_deg=0.0)
    assert frenet_right.d < 0.0
    assert abs(frenet_right.d) == pytest.approx(0.4, abs=1e-2)


def test_generate_track_boundaries() -> None:
    """Test parallel boundary offset generator helper function."""
    waypoints = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    left_b, right_b = generate_track_boundaries(waypoints, track_width=2.0)

    assert len(left_b) == len(waypoints)
    assert len(right_b) == len(waypoints)
