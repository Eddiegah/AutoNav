"""
Headless smoke test: runs 50 sim ticks of the full pipeline
(mock world → sensors → VO → mapping → A* → controller)
with no windows. Confirms end-to-end wiring is correct.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Suppress pygame display
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import mock_carla as carla
import math, numpy as np
from sensors         import SensorManager, build_camera_matrix
from visual_odometry import VisualOdometry, evaluate_drift
from mapping         import OccupancyGrid
from path_planning   import astar
from controller      import PurePursuitController

client  = carla.Client()
world   = client.get_world()
settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp_lib   = world.get_blueprint_library()
car_bp   = bp_lib.filter("vehicle.tesla.model3")[0]
spawn_pt = world.get_map().get_spawn_points()[0]
vehicle  = world.spawn_actor(car_bp, spawn_pt)

sensors  = SensorManager(world, vehicle)
K        = build_camera_matrix()
vo       = VisualOdometry(K)
spawn_x  = float(spawn_pt.location.x)
spawn_y  = float(spawn_pt.location.y)
occ      = OccupancyGrid(origin_x=spawn_x, origin_y=spawn_y)
ctrl     = PurePursuitController()

goal_world   = (80.0, 120.0)
gt_positions = []
est_positions = []
prev_gt = None

for tick in range(50):
    world.tick()
    tf    = vehicle.get_transform()
    gt    = (float(tf.location.x), float(tf.location.y))
    gt_yaw = math.radians(tf.rotation.yaw)
    speed = math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)

    gt_scale = None
    if prev_gt:
        gt_scale = math.hypot(gt[0]-prev_gt[0], gt[1]-prev_gt[1])
    prev_gt = gt

    cam = sensors.get_camera_frame()
    if cam is not None:
        pose    = vo.update(cam, gt_scale=gt_scale)
        est_pos = pose.position_2d
    else:
        est_pos = gt

    gt_positions.append(gt)
    est_positions.append(est_pos)

    lidar = sensors.get_lidar_points()
    occ.update(lidar, gt, gt_yaw)

    if tick % 20 == 0 or not ctrl.has_path():
        path = astar(occ, gt, goal_world)
        if path:
            ctrl.set_path(path)

    if ctrl.has_path():
        vc = ctrl.update(tf, speed, timestamp=float(tick)*0.05)
        vehicle.apply_control(vc)

sensors.destroy()

d = evaluate_drift(est_positions, gt_positions)
print(f"Smoke test passed — 50 ticks")
print(f"  VO frames : {vo.frame_count}")
print(f"  RMSE      : {d['rmse']:.4f} m")
print(f"  Final pos : {gt_positions[-1][0]:.2f}, {gt_positions[-1][1]:.2f}")
print(f"  Occ cells > 0.5: {int((occ.get_probability() > 0.5).sum())}")
print("All OK.")
