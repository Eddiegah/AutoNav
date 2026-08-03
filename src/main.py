"""
main.py — AutoNav entry point.

Wires the full pipeline together:
  SensorManager → VisualOdometry → OccupancyGrid → AStarPlanner
                                                  → PurePursuitController
                                                  → Dashboard

Two-process model:
  Process 1: CARLA server (CarlaUE4.exe / CarlaUE5.exe) — start this first.
  Process 2: This script — run with:  py -3.11 src/main.py

Usage:
    py -3.11 src/main.py [--host 127.0.0.1] [--port 2000]
                         [--goal-x 50] [--goal-y 80]
                         [--map Town01]
"""

from __future__ import annotations
import argparse
import math
import sys
import time
import traceback
import os

import carla
import numpy as np
import cv2

# ── Ensure src/ is on the Python path when run from project root ──────
sys.path.insert(0, os.path.dirname(__file__))

from sensors        import SensorManager, build_camera_matrix
from visual_odometry import VisualOdometry, evaluate_drift
from mapping        import OccupancyGrid
from path_planning  import astar
from controller     import PurePursuitController
from visualization  import Dashboard


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST    = "127.0.0.1"
DEFAULT_PORT    = 2000
DEFAULT_TIMEOUT = 10.0   # seconds to wait for CARLA connection
REPLAN_EVERY    = 30     # re-run A* every N ticks
FIXED_DELTA_S   = 0.05   # 20 Hz simulation step


# ---------------------------------------------------------------------------
# Helper: get speed in m/s from a CARLA actor
# ---------------------------------------------------------------------------

def _get_speed(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def _carla_pos_to_xy(location: carla.Location) -> tuple[float, float]:
    """Convert CARLA Location to our 2-D (x, y) world-plane convention."""
    return float(location.x), float(location.y)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    # ── Connect to CARLA ─────────────────────────────────────────────
    print(f"[AutoNav] Connecting to CARLA at {args.host}:{args.port} ...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(DEFAULT_TIMEOUT)
        world  = client.get_world()
    except RuntimeError as exc:
        print(
            "\n[AutoNav] ERROR: Could not connect to the CARLA server.\n"
            "  Make sure CarlaUE4.exe (or CarlaUE5.exe) is running BEFORE\n"
            "  starting this script.  See README → Setup → Two-process model.\n"
            f"  Detail: {exc}"
        )
        sys.exit(1)

    print(f"[AutoNav] Connected. Map: {world.get_map().name}")

    # ── Load map if requested ─────────────────────────────────────────
    if args.map:
        print(f"[AutoNav] Loading map {args.map} ...")
        client.load_world(args.map)
        world = client.get_world()
        time.sleep(2.0)   # give the world time to load

    # ── Synchronous mode (deterministic, reproducible) ───────────────
    settings = world.get_settings()
    settings.synchronous_mode     = True
    settings.fixed_delta_seconds  = FIXED_DELTA_S
    world.apply_settings(settings)
    print("[AutoNav] Synchronous mode enabled (20 Hz).")

    actor_list: list[carla.Actor] = []
    sensors: SensorManager | None = None
    dashboard: Dashboard | None   = None

    try:
        # ── Spawn vehicle ────────────────────────────────────────────
        bp_lib   = world.get_blueprint_library()
        car_bp   = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_pt = world.get_map().get_spawn_points()[0]
        vehicle  = world.spawn_actor(car_bp, spawn_pt)
        actor_list.append(vehicle)
        print(f"[AutoNav] Vehicle spawned: {vehicle.type_id} (id={vehicle.id})")

        # ── Attach sensors ────────────────────────────────────────────
        sensors = SensorManager(world, vehicle)

        # ── Initialise pipeline modules ───────────────────────────────
        K   = build_camera_matrix()
        vo  = VisualOdometry(K)
        # Centre grid on spawn point so world coords land inside the grid
        occ = OccupancyGrid(origin_x=float(spawn_pt.location.x),
                            origin_y=float(spawn_pt.location.y))
        ctrl = PurePursuitController()
        dashboard = Dashboard()

        # Trajectory buffers for drift evaluation
        gt_positions:  list[tuple[float, float]] = []
        est_positions: list[tuple[float, float]] = []

        # Goal in world frame
        goal_world = (float(args.goal_x), float(args.goal_y))
        print(f"[AutoNav] Navigation goal: {goal_world}")

        # ── Initial plan ──────────────────────────────────────────────
        # We can't plan yet (map is empty) — do first plan after N ticks
        planned_path: list[tuple[float, float]] | None = None
        tick_count = 0
        prev_gt_pos: tuple[float, float] | None = None

        print("[AutoNav] Starting main loop.  Press Ctrl-C to stop.")
        print("[AutoNav] Dashboard window: press Q inside it to stop cleanly.")

        while True:
            world.tick()   # advance simulation by one fixed step
            tick_count += 1

            # ── Ground truth from CARLA ───────────────────────────────
            tf       = vehicle.get_transform()
            gt_pos   = _carla_pos_to_xy(tf.location)
            gt_yaw   = math.radians(tf.rotation.yaw)
            speed    = _get_speed(vehicle)

            # ── Scale for VO from ground truth displacement ───────────
            gt_scale = None
            if prev_gt_pos is not None:
                gt_scale = math.hypot(gt_pos[0] - prev_gt_pos[0],
                                      gt_pos[1] - prev_gt_pos[1])
            prev_gt_pos = gt_pos

            # ── Visual odometry ───────────────────────────────────────
            cam_frame = sensors.get_camera_frame()
            if cam_frame is not None:
                pose = vo.update(cam_frame, gt_scale=gt_scale)
                est_pos = pose.position_2d
            else:
                est_pos = gt_pos   # fall back to GT until camera warms up

            gt_positions.append(gt_pos)
            est_positions.append(est_pos)

            # ── Occupancy grid update ─────────────────────────────────
            lidar_pts = sensors.get_lidar_points()
            occ.update(lidar_pts, gt_pos, gt_yaw)

            # ── (Re-)plan path ────────────────────────────────────────
            if tick_count % REPLAN_EVERY == 0 or planned_path is None:
                new_path = astar(occ, gt_pos, goal_world)
                if new_path is not None:
                    planned_path = new_path
                    ctrl.set_path(planned_path)
                    print(f"[AutoNav] Path re-planned: {len(planned_path)} waypoints")
                elif planned_path is None:
                    print("[AutoNav] WARNING: No path found yet (map may be empty)")

            # ── Vehicle control ───────────────────────────────────────
            if ctrl.has_path():
                sim_time = world.get_snapshot().timestamp.elapsed_seconds
                vc = ctrl.update(tf, speed, timestamp=sim_time)
                vehicle.apply_control(vc)

                if ctrl.path_complete(gt_pos):
                    print("[AutoNav] Goal reached!")
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    break
            else:
                # No path yet — idle
                vehicle.apply_control(carla.VehicleControl(throttle=0.0,
                                                            brake=1.0))

            # ── Drift evaluation (every 10 ticks) ────────────────────
            if tick_count % 10 == 0 and len(gt_positions) > 5:
                drift = evaluate_drift(est_positions, gt_positions)
                dashboard.rmse        = drift["rmse"]
                dashboard.final_drift = drift["final_drift"]

            # ── Dashboard ─────────────────────────────────────────────
            dashboard.update(
                camera_bgr      = cam_frame,
                occ_grid        = occ,
                vehicle_pos_gt  = gt_pos,
                vehicle_pos_est = est_pos,
                planned_path    = planned_path,
                vehicle_yaw_rad = gt_yaw,
                speed_mps       = speed,
                lookahead_point = ctrl.lookahead_point,
            )

            # Check for Q key press in OpenCV window
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # Q or Escape
                print("[AutoNav] User requested stop.")
                break

        # ── Final drift report ────────────────────────────────────────
        print("\n── Visual Odometry Drift Report ──────────────────────────")
        if len(gt_positions) > 1:
            drift = evaluate_drift(est_positions, gt_positions)
            print(f"  Frames processed : {vo.frame_count}")
            print(f"  Frames skipped   : {vo.skipped_frames}")
            print(f"  RMSE             : {drift['rmse']:.3f} m")
            print(f"  Max error        : {drift['max_error']:.3f} m")
            print(f"  Final drift      : {drift['final_drift']:.3f} m")
            print(
                "\n  NOTE: Drift is the honest cost of monocular visual odometry\n"
                "  without loop closure.  Larger drifts on longer drives are\n"
                "  expected — see README for explanation and future work.\n"
            )
        else:
            print("  (Not enough data collected)")

        # ── Save trajectory plot ──────────────────────────────────────
        os.makedirs("results", exist_ok=True)
        dashboard.save_trajectory_plot("results/trajectory_comparison.png")

    except KeyboardInterrupt:
        print("\n[AutoNav] Interrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        # ── Teardown ──────────────────────────────────────────────────
        print("[AutoNav] Cleaning up ...")
        if sensors:
            sensors.destroy()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        # Restore async mode so CARLA doesn't hang after we exit
        try:
            settings = world.get_settings()
            settings.synchronous_mode    = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass
        if dashboard:
            dashboard.close()
        print("[AutoNav] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoNav — CARLA navigation stack")
    parser.add_argument("--host",   default=DEFAULT_HOST, help="CARLA server host")
    parser.add_argument("--port",   type=int, default=DEFAULT_PORT,
                        help="CARLA server port")
    parser.add_argument("--goal-x", type=float, default=50.0,
                        help="Goal X coordinate in world frame (metres)")
    parser.add_argument("--goal-y", type=float, default=80.0,
                        help="Goal Y coordinate in world frame (metres)")
    parser.add_argument("--map",    default="Town01",
                        help="CARLA map name (e.g. Town01, Town03)")
    args = parser.parse_args()
    run(args)
