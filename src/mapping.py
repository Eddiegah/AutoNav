"""
mapping.py — Real-time 2-D occupancy grid from LiDAR point clouds.

────────────────────────────────────────────────────────────────
HOW THE OCCUPANCY GRID WORKS
────────────────────────────────────────────────────────────────
An occupancy grid divides the world plane into a regular grid of
cells.  Each cell stores a log-odds value:

  log_odds > 0  →  cell is likely OCCUPIED  (obstacle)
  log_odds < 0  →  cell is likely FREE       (driveable)
  log_odds = 0  →  unknown

Using log-odds (instead of raw probabilities) makes incremental
updates numerically stable and lets us clamp the value to a finite
range without losing precision at the extremes.

Update rule (Bayesian occupancy update in log-odds form):
  L(t) = L(t-1) + log_odds_hit    if a LiDAR point lands in the cell
  L(t) = L(t-1) + log_odds_free   if the ray *passes through* the cell
  L(t) = clamp(L(t), L_min, L_max)

We project each LiDAR point into the 2-D world plane (drop the z
axis), look up its grid cell, and mark it as occupied.  We also
trace the ray from the sensor origin to that point and mark all
intermediate cells as free (Bresenham line algorithm).

This is the standard technique used in ROS's gmapping and costmap_2d.
It is *not* full SLAM — we have no loop closure.  Pose errors from
visual odometry accumulate directly into the map (cells may be
marked occupied in the wrong place).  This is a known limitation;
see README.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

GRID_RESOLUTION = 0.4     # metres per cell  (0.4 m = good balance of detail vs speed)
GRID_SIZE_M     = 260.0   # physical size of the grid square (metres) — covers mock world
GRID_CELLS      = int(GRID_SIZE_M / GRID_RESOLUTION)   # cells per side

# Log-odds update values
LO_HIT   =  0.85   # log-odds added when a point lands in a cell
LO_FREE  = -0.40   # log-odds added when a ray passes through a cell
LO_MIN   = -5.0    # clamp lower bound (very confident free)
LO_MAX   =  5.0    # clamp upper bound (very confident occupied)

# Threshold for binary visualisation
OCCUPIED_THRESH  =  0.5
FREE_THRESH      = -0.5

# Height filter: only include LiDAR points between these z-values
# (relative to the sensor mount).  Keep generous bounds for the mock.
Z_MIN = -2.0   # metres (relative to the sensor)
Z_MAX =  4.0


class OccupancyGrid:
    """
    A fixed-size 2-D occupancy grid centred on the vehicle's starting
    position.

    The grid is stored as a (GRID_CELLS × GRID_CELLS) float32 array
    of log-odds values.  Index [row, col] corresponds to world
    coordinate:
        world_x = (col - GRID_CELLS/2) * GRID_RESOLUTION
        world_z = (row - GRID_CELLS/2) * GRID_RESOLUTION
    (CARLA uses x=forward, z=up, y=right; our 2-D plane is x/y
    in the navigator's frame, which maps to CARLA's x/z.)
    """

    def __init__(self, origin_x: float = 0.0, origin_y: float = 0.0):
        self._grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
        self._origin_offset = GRID_CELLS // 2   # cell index of world origin
        # World coordinates of the grid centre
        self._origin_x = origin_x
        self._origin_y = origin_y

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self,
               lidar_points: np.ndarray,
               vehicle_pos_xz: tuple[float, float],
               vehicle_yaw_rad: float = 0.0) -> None:
        """
        Incorporate a new LiDAR sweep into the grid.

        Parameters
        ----------
        lidar_points : np.ndarray  shape (N, 4)  [x, y, z, intensity]
            Point cloud in the *sensor* frame (CARLA convention:
            x forward, y right, z up).
        vehicle_pos_xz : (x, z)
            Current vehicle position in the *world* frame (metres).
            Comes from CARLA ground truth or VO estimate.
        vehicle_yaw_rad : float
            Vehicle heading (radians, counter-clockwise from +x axis).
        """
        if lidar_points is None or len(lidar_points) == 0:
            return

        # ── Height filter ────────────────────────────────────────────
        pts = lidar_points
        z_vals = pts[:, 2]
        mask   = (z_vals > Z_MIN) & (z_vals < Z_MAX)
        pts    = pts[mask]
        if len(pts) == 0:
            return

        # ── Rotate points from sensor frame to world frame ────────────
        # Sensor frame: x forward, y right → world frame: same convention
        # (just translate + rotate by vehicle yaw).
        cos_y, sin_y = np.cos(vehicle_yaw_rad), np.sin(vehicle_yaw_rad)
        # 2-D rotation of (x, y) columns
        px = pts[:, 0]
        py = pts[:, 1]
        world_x = cos_y * px - sin_y * py + vehicle_pos_xz[0]
        world_y = sin_y * px + cos_y * py + vehicle_pos_xz[1]

        # Sensor origin in world frame
        ox, oy = vehicle_pos_xz

        # ── Convert world coords → grid indices ───────────────────────
        def world_to_cell(wx, wy):
            col = ((wx - self._origin_x) / GRID_RESOLUTION + self._origin_offset).astype(int)
            row = ((wy - self._origin_y) / GRID_RESOLUTION + self._origin_offset).astype(int)
            return row, col

        sensor_row = int((oy - self._origin_y) / GRID_RESOLUTION + self._origin_offset)
        sensor_col = int((ox - self._origin_x) / GRID_RESOLUTION + self._origin_offset)

        rows, cols = world_to_cell(world_x, world_y)

        # ── Bresenham ray-tracing + log-odds update ───────────────────
        valid = (
            (rows >= 0) & (rows < GRID_CELLS) &
            (cols >= 0) & (cols < GRID_CELLS)
        )
        for r, c in zip(rows[valid], cols[valid]):
            # Trace free cells along the ray
            for fr, fc in _bresenham(sensor_row, sensor_col, r, c):
                self._grid[fr, fc] = np.clip(
                    self._grid[fr, fc] + LO_FREE, LO_MIN, LO_MAX)
            # Mark the endpoint as occupied
            self._grid[r, c] = np.clip(
                self._grid[r, c] + LO_HIT, LO_MIN, LO_MAX)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_log_odds(self) -> np.ndarray:
        """Return the raw log-odds grid (GRID_CELLS × GRID_CELLS)."""
        return self._grid

    def get_binary(self) -> np.ndarray:
        """
        Return a uint8 grid:
          255 = occupied
            0 = free / unknown
        """
        binary = np.zeros_like(self._grid, dtype=np.uint8)
        binary[self._grid > OCCUPIED_THRESH] = 255
        return binary

    def get_probability(self) -> np.ndarray:
        """
        Return occupancy probability P(occupied) in [0, 1] for display.
        P = exp(L) / (1 + exp(L))  — the sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-self._grid))

    def is_occupied(self, world_x: float, world_y: float) -> bool:
        """Query whether a world-frame point is currently marked occupied."""
        col = int((world_x - self._origin_x) / GRID_RESOLUTION + self._origin_offset)
        row = int((world_y - self._origin_y) / GRID_RESOLUTION + self._origin_offset)
        if 0 <= row < GRID_CELLS and 0 <= col < GRID_CELLS:
            return bool(self._grid[row, col] > OCCUPIED_THRESH)
        return True  # treat out-of-bounds as occupied (conservative)

    def world_to_cell(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert world coordinates (metres) to (row, col) grid indices."""
        col = int((world_x - self._origin_x) / GRID_RESOLUTION + self._origin_offset)
        row = int((world_y - self._origin_y) / GRID_RESOLUTION + self._origin_offset)
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert (row, col) grid indices back to world coordinates (metres)."""
        wx = (col - self._origin_offset) * GRID_RESOLUTION + self._origin_x
        wy = (row - self._origin_offset) * GRID_RESOLUTION + self._origin_y
        return wx, wy

    def reset(self):
        """Zero the grid (call when teleporting the vehicle)."""
        self._grid[:] = 0.0


# ---------------------------------------------------------------------------
# Bresenham line algorithm
# ---------------------------------------------------------------------------

def _bresenham(r0: int, c0: int, r1: int, c1: int):
    """
    Yield (row, col) integer cell indices along the line from
    (r0, c0) to (r1, c1), *excluding* the endpoint.

    This is the standard integer Bresenham algorithm — O(max(|dr|,|dc|))
    and allocation-free (generator).
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0

    while True:
        if r == r1 and c == c1:
            break
        # bounds check — stop tracing if we leave the grid
        if not (0 <= r < GRID_CELLS and 0 <= c < GRID_CELLS):
            break
        yield r, c
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r   += sr
        if e2 < dr:
            err += dr
            c   += sc
