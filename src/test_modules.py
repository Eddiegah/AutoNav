"""Offline test of all non-CARLA modules. Run: venv\Scripts\python test_modules.py"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np

# ── 1. sensors.py ────────────────────────────────────────────────────
from sensors import build_camera_matrix
K = build_camera_matrix()
assert K.shape == (3, 3)
print(f"[sensors]  camera matrix OK  fx={K[0,0]:.1f}")

# ── 2. visual_odometry.py ────────────────────────────────────────────
from visual_odometry import VisualOdometry, evaluate_drift
vo = VisualOdometry(K)

# Two synthetic checkerboard frames
tile = np.kron([[0,255]*4]*4 + [[255,0]*4]*4, np.ones((75,100), dtype=np.uint8))
frame1 = np.stack([tile.astype(np.uint8)]*3, axis=-1)
frame2 = np.roll(frame1, 5, axis=1)   # 5-px horizontal shift

vo.update(frame1)
pose2 = vo.update(frame2)
print(f"[vo]       update OK  t={pose2.t.ravel()}")

est = [(0,0),(1,0),(2,0.1)]
gt  = [(0,0),(1,0),(2,0  )]
d   = evaluate_drift(est, gt)
rmse_str = f"{d['rmse']:.4f}"
print(f"[vo]       drift eval OK  RMSE={rmse_str}")

# ── 3. mapping.py ────────────────────────────────────────────────────
from mapping import OccupancyGrid
occ = OccupancyGrid()  # origin (0,0) fine for unit test
angles = np.linspace(-1.0, 1.0, 100)
pts = np.column_stack([
    10 * np.cos(angles),
    10 * np.sin(angles),
    0.5 * np.ones(100),
    np.ones(100),
]).astype(np.float32)
occ.update(pts, (0.0, 0.0), 0.0)
prob = occ.get_probability()
maxp = f"{prob.max():.3f}"
print(f"[mapping]  update OK  shape={prob.shape}  max_prob={maxp}")

# ── 4. path_planning.py ──────────────────────────────────────────────
from path_planning import astar

# Empty grid — straight path
occ2 = OccupancyGrid()
path = astar(occ2, (0.0, 0.0), (10.0, 10.0))
assert path is not None, "A* returned None on empty grid"
print(f"[planner]  A* empty grid OK  {len(path)} waypoints")

# Grid with obstacle wall blocking direct path
occ3 = OccupancyGrid()
obs_pts = np.column_stack([
    5.0 * np.ones(30),
    np.linspace(-4.0, 4.0, 30),
    0.5 * np.ones(30),
    np.ones(30),
]).astype(np.float32)
for _ in range(20):
    occ3.update(obs_pts, (0.0, 0.0), 0.0)
path2 = astar(occ3, (0.0, 0.0), (15.0, 0.0))
result = "path found" if path2 else "no path"
wpts   = len(path2) if path2 else 0
print(f"[planner]  A* with obstacle: {result}  ({wpts} wpts)")

# ── 5. controller.py ─────────────────────────────────────────────────
from controller import PurePursuitController
ctrl = PurePursuitController()
ctrl.set_path([(i * 2.0, 0.0) for i in range(20)])
assert ctrl.has_path()

class FakeLoc:
    x = y = z = 0.0
class FakeRot:
    yaw = 0.0
class FakeTF:
    location = FakeLoc()
    rotation = FakeRot()

vc = ctrl.update(FakeTF(), current_speed_mps=3.0, timestamp=0.0)
steer_str    = f"{vc.steer:.3f}"
throttle_str = f"{vc.throttle:.3f}"
print(f"[ctrl]     Pure Pursuit OK  steer={steer_str}  throttle={throttle_str}")

# ── 6. visualization.py (no window — just tile construction) ─────────
from visualization import Dashboard, _draw_diamond
import cv2

dash = Dashboard.__new__(Dashboard)  # skip cv2.namedWindow
dash._gt_traj   = []
dash._est_traj  = []
dash.rmse        = 0.0
dash.final_drift = 0.0
dash._tick       = 0

# Map tile
occ4 = OccupancyGrid()
occ4.update(pts, (0.0, 0.0), 0.0)
map_tile = dash._map_panel(occ4, (0.0,0.0), (1.0,1.0), None, 0.0, None)
assert map_tile.shape[2] == 3
print(f"[viz]      map tile OK  shape={map_tile.shape}")

# Camera tile
cam_tile = dash._camera_panel(frame1, 5.0)
print(f"[viz]      camera tile OK  shape={cam_tile.shape}")

# Trajectory tile
dash._gt_traj  = [(0,0),(5,5),(10,3)]
dash._est_traj = [(0,0),(5.2,4.8),(10.3,2.9)]
traj_tile = dash._trajectory_panel()
print(f"[viz]      trajectory tile OK  shape={traj_tile.shape}")

print()
print("=" * 45)
print("All module tests passed.")
print("=" * 45)
