"""
controller.py — Pure Pursuit path-following controller.

────────────────────────────────────────────────────────────────
PURE PURSUIT ALGORITHM
────────────────────────────────────────────────────────────────
Pure Pursuit is a classical geometric path-tracking algorithm
developed at Carnegie Mellon in the late 1980s and still widely
used because of its simplicity and smooth behaviour.

Key idea:
  Imagine the vehicle is "chasing" a *lookahead point* — a point
  on the planned path some fixed distance (Ld) ahead of the
  vehicle's current rear-axle position.  The steering angle is
  chosen so that the vehicle follows a circular arc from its
  current position to that lookahead point.

Maths in brief:
  1. Find the lookahead point P_L on the path at distance Ld ahead.
  2. Compute α = angle between the vehicle heading and the line to P_L.
  3. Steering angle δ = atan2(2 * L * sin(α), Ld)
     where L is the vehicle wheelbase.
  4. Apply δ as the CARLA steer command (normalised to [-1, 1]).

A PID speed controller maintains the desired cruise speed.

Tuning tips:
  • Larger Ld → smoother path following, more "cutting corners".
  • Smaller Ld → tighter tracking, more oscillation at speed.
  • Ld can be made adaptive: Ld = k * v  (scales with speed).
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import math
import numpy as np
try:
    import carla
except ImportError:
    import mock_carla as carla  # type: ignore


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

WHEELBASE_M = 2.9          # approximate wheelbase (metres) for a sedan
MAX_STEER   = 1.0          # CARLA steer command range [-1, 1]

# Pure Pursuit lookahead
LOOKAHEAD_BASE_M  = 4.0    # base lookahead distance (metres)
LOOKAHEAD_K       = 0.5    # adaptive factor: Ld = base + k * speed

# Speed control (PID)
TARGET_SPEED_MPS  = 5.0    # ~18 km/h — safe for obstacle-dense environments
KP_SPEED          = 0.3
KI_SPEED          = 0.01
KD_SPEED          = 0.05

# Braking: apply brakes if we need to slow significantly
BRAKE_THRESHOLD   = -0.5   # throttle < this → add brake instead


class PurePursuitController:
    """
    Translates a sequence of 2-D waypoints into CARLA VehicleControl
    commands (steer, throttle, brake).

    Call set_path() to upload a new path, then call update() every
    simulation tick with the current vehicle transform.
    """

    def __init__(self,
                 target_speed: float = TARGET_SPEED_MPS,
                 wheelbase:    float = WHEELBASE_M):
        self._target_speed = target_speed
        self._wheelbase    = wheelbase

        # Planned path as list of (x, y) world-frame points
        self._path: list[tuple[float, float]] = []
        self._path_idx: int = 0   # index of the most recent waypoint we passed

        # PID state for speed control
        self._integral_err: float  = 0.0
        self._prev_err:     float  = 0.0
        self._prev_time:    float | None = None

        # For dashboard: expose these so visualisation.py can read them
        self.lookahead_point: tuple[float, float] | None = None
        self.cross_track_error: float = 0.0

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def set_path(self, waypoints: list[tuple[float, float]]):
        """
        Upload a new planned path.  The vehicle will start tracking
        from the beginning of the list.
        """
        self._path     = waypoints
        self._path_idx = 0
        self.lookahead_point = None

    def has_path(self) -> bool:
        return len(self._path) > 0

    def path_complete(self, vehicle_pos: tuple[float, float],
                      tolerance_m: float = 3.0) -> bool:
        """Return True when the vehicle is within *tolerance_m* of the goal."""
        if not self._path:
            return False
        gx, gy = self._path[-1]
        vx, vy = vehicle_pos
        return math.hypot(gx - vx, gy - vy) < tolerance_m

    # ------------------------------------------------------------------
    # Main control update
    # ------------------------------------------------------------------

    def update(self,
               transform: carla.Transform,
               current_speed_mps: float,
               timestamp: float | None = None
               ) -> carla.VehicleControl:
        """
        Compute and return a VehicleControl for the current tick.

        Parameters
        ----------
        transform : carla.Transform
            Current vehicle world transform (from actor.get_transform()).
        current_speed_mps : float
            Current vehicle speed in m/s.
        timestamp : float, optional
            Simulation time in seconds (for PID derivative term).

        Returns
        -------
        carla.VehicleControl
        """
        ctrl = carla.VehicleControl()

        if not self._path:
            # No path — coast to a stop
            ctrl.brake    = 1.0
            ctrl.throttle = 0.0
            return ctrl

        vx = transform.location.x
        vy = transform.location.y
        vz = transform.location.z

        # CARLA yaw: degrees, 0 = east (+x), positive counter-clockwise
        yaw_rad = math.radians(transform.rotation.yaw)

        # ── 1. Adaptive lookahead distance ────────────────────────────
        Ld = LOOKAHEAD_BASE_M + LOOKAHEAD_K * current_speed_mps

        # ── 2. Find lookahead point on path ───────────────────────────
        lp = self._find_lookahead(vx, vy, Ld)
        self.lookahead_point = lp

        if lp is None:
            # Reached end of path — brake
            ctrl.brake    = 1.0
            ctrl.throttle = 0.0
            return ctrl

        # ── 3. Compute α (heading error to lookahead point) ───────────
        dx = lp[0] - vx
        dy = lp[1] - vy
        angle_to_lp = math.atan2(dy, dx)
        alpha = angle_to_lp - yaw_rad
        # Normalise to [-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Cross-track error (signed lateral deviation, for dashboard)
        self.cross_track_error = Ld * math.sin(alpha)

        # ── 4. Pure Pursuit steering formula ─────────────────────────
        # δ = atan(2 * L * sin(α) / Ld)
        steer_rad  = math.atan2(2.0 * self._wheelbase * math.sin(alpha), Ld)
        # CARLA expects steer in [-1, 1]; max physical steer ≈ 70°
        steer_norm = np.clip(steer_rad / math.radians(70.0), -1.0, 1.0)
        ctrl.steer = float(steer_norm)

        # ── 5. PID speed controller ───────────────────────────────────
        throttle = self._speed_pid(current_speed_mps, timestamp)
        if throttle >= 0:
            ctrl.throttle = float(np.clip(throttle, 0.0, 1.0))
            ctrl.brake    = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake    = float(np.clip(-throttle, 0.0, 1.0))

        return ctrl

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_lookahead(self,
                        vx: float, vy: float,
                        Ld: float) -> tuple[float, float] | None:
        """
        Walk forward along the path from _path_idx and return the first
        waypoint at least Ld metres ahead of (vx, vy).

        Advances _path_idx so we never backtrack.
        """
        if not self._path:
            return None

        # Advance past waypoints we've already passed (within 1 m)
        while self._path_idx < len(self._path) - 1:
            wx, wy = self._path[self._path_idx]
            if math.hypot(wx - vx, wy - vy) < 1.0:
                self._path_idx += 1
            else:
                break

        # Find the first point at distance ≥ Ld
        for i in range(self._path_idx, len(self._path)):
            wx, wy = self._path[i]
            if math.hypot(wx - vx, wy - vy) >= Ld:
                return wx, wy

        # Past the end — return the final waypoint
        return self._path[-1] if self._path else None

    def _speed_pid(self, current_speed: float,
                   timestamp: float | None) -> float:
        """
        PID controller for longitudinal speed.
        Returns a signed throttle value (negative → need to brake).
        """
        err = self._target_speed - current_speed
        dt  = 0.05   # default dt if timestamp not available

        if timestamp is not None and self._prev_time is not None:
            dt = max(timestamp - self._prev_time, 1e-4)
        self._prev_time = timestamp

        self._integral_err += err * dt
        # Anti-windup: clamp integral term
        self._integral_err = np.clip(self._integral_err, -10.0, 10.0)

        d_err = (err - self._prev_err) / dt if self._prev_err is not None else 0.0
        self._prev_err = err

        output = KP_SPEED * err + KI_SPEED * self._integral_err + KD_SPEED * d_err
        return float(output)
