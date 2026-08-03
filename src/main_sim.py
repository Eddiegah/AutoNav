"""
main_sim.py — AutoNav with the built-in mock simulator (no CARLA needed).

Runs the full pipeline:
  mock physics vehicle → sensors → VO → occupancy grid → A* → Pure Pursuit

Two windows open:
  1. AutoNav Dashboard  (OpenCV) — camera / map / trajectory tiles
  2. SimView            (pygame) — top-down bird's-eye view of the world

Usage:
    venv\\Scripts\\python src\\main_sim.py [--goal-x 80] [--goal-y 120]

Press Q in either window (or Ctrl-C in terminal) to stop.
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import traceback

# ── Put src/ on path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Use mock_carla instead of real carla ─────────────────────────────
import mock_carla as carla

import numpy as np
import cv2
import pygame

from sensors         import SensorManager, build_camera_matrix
from visual_odometry import VisualOdometry, evaluate_drift
from mapping         import OccupancyGrid
from path_planning   import astar
from controller      import PurePursuitController
from visualization   import Dashboard
from mock_carla.world import OBSTACLES, WORLD_SIZE, ROAD_WIDTH, BLOCK_SIZE, N_BLOCKS


# ---------------------------------------------------------------------------
# Pygame top-down viewer  — dark city map aesthetic
# ---------------------------------------------------------------------------

SIM_WIN_SIZE = 720

# Colour palette (RGB for pygame)
_SKY_BG       = ( 18,  20,  24)
_ROAD_COL     = ( 42,  46,  52)
_ROAD_LINE    = ( 70,  76,  84)
_LANE_COL     = (160, 155, 100)
_BLDG_COLS    = [
    ( 65,  75,  90), ( 75,  68,  58), ( 60,  80,  72),
    ( 80,  70,  60), ( 70,  65,  80), ( 58,  76,  68),
]
_BLDG_ROOF    = ( 30,  35,  42)
_PATH_COL     = ( 40, 200, 140)
_GT_COL       = ( 60, 210,  80)
_EST_COL      = ( 80,  80, 240)
_GOAL_COL     = (255, 200,  40)
_VEH_COL      = ( 40, 220,  80)
_VEH_BODY     = ( 20, 160,  60)
_TEXT_COL     = (210, 215, 220)
_DIM_COL      = (100, 110, 120)
_ACCENT_COL   = ( 60, 180, 255)


class SimView:
    """Top-down city-map style bird's-eye renderer."""

    def __init__(self, world_size: float):
        pygame.init()
        self._screen    = pygame.display.set_mode((SIM_WIN_SIZE, SIM_WIN_SIZE))
        pygame.display.set_caption("AutoNav — Top-Down View")
        self._font_sm   = pygame.font.SysFont("consolas", 13)
        self._font_med  = pygame.font.SysFont("consolas", 16, bold=True)
        self._font_lg   = pygame.font.SysFont("consolas", 22, bold=True)
        self._scale     = SIM_WIN_SIZE / world_size
        self._world_size = world_size
        # Pre-render the static city background
        self._bg        = self._render_background()
        self._trail_surf = pygame.Surface((SIM_WIN_SIZE, SIM_WIN_SIZE), pygame.SRCALPHA)
        self._trail_surf.fill((0, 0, 0, 0))

    def _w2s(self, wx, wy):
        return int(wx * self._scale), int(wy * self._scale)

    def _render_background(self) -> pygame.Surface:
        """Draw the static city grid once — roads, kerbs, buildings."""
        surf = pygame.Surface((SIM_WIN_SIZE, SIM_WIN_SIZE))
        surf.fill(_SKY_BG)

        s = self._scale

        # ── Grass/ground fill ─────────────────────────────────────────
        surf.fill((28, 38, 28))

        # ── Road surface ──────────────────────────────────────────────
        hw = ROAD_WIDTH / 2
        for b in range(N_BLOCKS + 1):
            road_pos = b * BLOCK_SIZE
            # Horizontal road band
            r = pygame.Rect(0, int((road_pos - hw) * s),
                            SIM_WIN_SIZE, int(ROAD_WIDTH * s + 1))
            pygame.draw.rect(surf, _ROAD_COL, r)
            # Vertical road band
            r = pygame.Rect(int((road_pos - hw) * s), 0,
                            int(ROAD_WIDTH * s + 1), SIM_WIN_SIZE)
            pygame.draw.rect(surf, _ROAD_COL, r)

        # ── Lane centre-line dashes ───────────────────────────────────
        dash_len  = int(6 * s)
        dash_gap  = int(10 * s)
        dash_w    = max(1, int(0.25 * s))
        for b in range(N_BLOCKS + 1):
            road_pos = b * BLOCK_SIZE
            mid_px   = int(road_pos * s)

            # Horizontal dashes
            x = 0
            while x < SIM_WIN_SIZE:
                pygame.draw.rect(surf, _LANE_COL,
                                 (x, mid_px - dash_w // 2, dash_len, dash_w))
                x += dash_len + dash_gap

            # Vertical dashes
            y = 0
            while y < SIM_WIN_SIZE:
                pygame.draw.rect(surf, _LANE_COL,
                                 (mid_px - dash_w // 2, y, dash_w, dash_len))
                y += dash_len + dash_gap

        # ── Kerb lines ────────────────────────────────────────────────
        kerb_w = max(1, int(0.8 * s))
        for b in range(N_BLOCKS + 1):
            road_pos = b * BLOCK_SIZE
            for sign in (-1, 1):
                edge = road_pos + sign * hw
                px   = int(edge * s)
                pygame.draw.rect(surf, (140, 135, 125),
                                 (0, px - kerb_w // 2, SIM_WIN_SIZE, kerb_w))
                pygame.draw.rect(surf, (140, 135, 125),
                                 (px - kerb_w // 2, 0, kerb_w, SIM_WIN_SIZE))

        # ── Buildings ─────────────────────────────────────────────────
        for i, (cx, cy, hw_b, hh_b) in enumerate(OBSTACLES):
            col  = _BLDG_COLS[i % len(_BLDG_COLS)]
            rx   = int((cx - hw_b) * s)
            ry   = int((cy - hh_b) * s)
            rw   = int(hw_b * 2 * s)
            rh   = int(hh_b * 2 * s)
            # Body
            pygame.draw.rect(surf, col, (rx, ry, rw, rh))
            # Roof shade (darker top edge)
            roof_h = max(2, rh // 8)
            pygame.draw.rect(surf, _BLDG_ROOF, (rx, ry, rw, roof_h))
            # Window grid
            win_rows, win_cols = 4, 3
            ww = max(2, rw // (win_cols * 2 + 1))
            wh = max(2, rh // (win_rows * 2 + 1))
            for wr in range(win_rows):
                for wc in range(win_cols):
                    lit = (i * 11 + wr * 7 + wc * 3) % 5 != 0
                    win_col = (220, 200, 120) if lit else (20, 25, 35)
                    wx_ = rx + (2 * wc + 1) * rw // (win_cols * 2 + 1)
                    wy_ = ry + roof_h + (2 * wr + 1) * (rh - roof_h) // (win_rows * 2 + 1)
                    pygame.draw.rect(surf, win_col, (wx_, wy_, ww, wh))
            # Outline
            pygame.draw.rect(surf, _BLDG_ROOF, (rx, ry, rw, rh), 1)

        return surf

    def render(self, vehicle_pos, vehicle_yaw, path,
               gt_traj, est_traj, goal, speed, rmse) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False

        # Static background
        self._screen.blit(self._bg, (0, 0))

        # ── Goal beacon ───────────────────────────────────────────────
        gx, gy = self._w2s(*goal)
        # Pulsing ring using tick
        import time
        pulse = int(abs(math.sin(time.time() * 2.5)) * 8)
        for r, a in [(22 + pulse, 80), (14, 160), (7, 255)]:
            s_goal = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s_goal, (*_GOAL_COL, a), (r, r), r)
            self._screen.blit(s_goal, (gx - r, gy - r))
        pygame.draw.circle(self._screen, _GOAL_COL, (gx, gy), 5)

        # ── Planned path ──────────────────────────────────────────────
        if path and len(path) > 1:
            pts = [self._w2s(*p) for p in path]
            # Glow: thick translucent underline
            path_surf = pygame.Surface((SIM_WIN_SIZE, SIM_WIN_SIZE), pygame.SRCALPHA)
            pygame.draw.lines(path_surf, (*_PATH_COL, 60), False, pts, 5)
            self._screen.blit(path_surf, (0, 0))
            # Crisp line on top
            pygame.draw.lines(self._screen, _PATH_COL, False, pts, 2)
            # Waypoint dots every N points
            for i, pt in enumerate(pts):
                if i % 8 == 0:
                    pygame.draw.circle(self._screen, _PATH_COL, pt, 2)

        # ── Persistent trail (GT = green, EST = blue) ─────────────────
        if len(gt_traj) > 1:
            pts = [self._w2s(*p) for p in gt_traj[-500:]]
            pygame.draw.lines(self._screen, _GT_COL, False, pts, 2)
        if len(est_traj) > 1:
            pts = [self._w2s(*p) for p in est_traj[-500:]]
            pygame.draw.lines(self._screen, _EST_COL, False, pts, 1)

        # ── Vehicle body ──────────────────────────────────────────────
        vx, vy = self._w2s(*vehicle_pos)
        body_len = max(12, int(4.5 * self._scale))
        body_w   = max(7,  int(2.0 * self._scale))

        def _rot(pts_local, yaw, cx, cy):
            out = []
            for lx, ly in pts_local:
                rx = lx * math.cos(yaw) - ly * math.sin(yaw)
                ry = lx * math.sin(yaw) + ly * math.cos(yaw)
                out.append((int(cx + rx), int(cy + ry)))
            return out

        # Car body rectangle
        body_pts = _rot([
            ( body_len // 2,  body_w // 2),
            (-body_len // 2,  body_w // 2),
            (-body_len // 2, -body_w // 2),
            ( body_len // 2, -body_w // 2),
        ], vehicle_yaw, vx, vy)
        pygame.draw.polygon(self._screen, _VEH_BODY, body_pts)
        pygame.draw.polygon(self._screen, _VEH_COL,  body_pts, 1)

        # Headlights
        for sign in (1, -1):
            hl = _rot([(body_len//2, sign * body_w//3)], vehicle_yaw, vx, vy)[0]
            pygame.draw.circle(self._screen, (255, 240, 180), hl, 2)

        # Direction arrow
        fwd = (int(vx + (body_len * 0.7) * math.cos(vehicle_yaw)),
               int(vy + (body_len * 0.7) * math.sin(vehicle_yaw)))
        pygame.draw.line(self._screen, (255, 255, 255), (vx, vy), fwd, 2)

        # ── HUD panel (bottom strip) ──────────────────────────────────
        hud_h = 72
        hud_surf = pygame.Surface((SIM_WIN_SIZE, hud_h), pygame.SRCALPHA)
        hud_surf.fill((15, 18, 22, 210))
        self._screen.blit(hud_surf, (0, SIM_WIN_SIZE - hud_h))
        pygame.draw.line(self._screen, (55, 65, 75),
                         (0, SIM_WIN_SIZE - hud_h),
                         (SIM_WIN_SIZE, SIM_WIN_SIZE - hud_h), 1)

        y0 = SIM_WIN_SIZE - hud_h + 12
        # Speed
        spd_surf = self._font_lg.render(f"{speed*3.6:5.1f}", True, _ACCENT_COL)
        self._screen.blit(spd_surf, (16, y0))
        u_surf = self._font_sm.render("km/h", True, _DIM_COL)
        self._screen.blit(u_surf, (92, y0 + 18))

        # Drift
        drift_col = _GT_COL if rmse < 3 else (_GOAL_COL if rmse < 8 else (220, 60, 60))
        d_surf  = self._font_med.render(f"VO drift  {rmse:.2f} m", True, drift_col)
        self._screen.blit(d_surf, (160, y0 + 4))

        # Goal distance
        dx_ = goal[0] - vehicle_pos[0]
        dy_ = goal[1] - vehicle_pos[1]
        dist_to_goal = math.hypot(dx_, dy_)
        g_surf = self._font_med.render(f"Goal  {dist_to_goal:.1f} m", True, _GOAL_COL)
        self._screen.blit(g_surf, (160, y0 + 28))

        # Key hint
        k_surf = self._font_sm.render("Q  quit", True, _DIM_COL)
        self._screen.blit(k_surf, (SIM_WIN_SIZE - 80, y0 + 24))

        # ── Mini compass ──────────────────────────────────────────────
        cx_, cy_ = SIM_WIN_SIZE - 45, SIM_WIN_SIZE - hud_h - 45
        pygame.draw.circle(self._screen, (30, 35, 42), (cx_, cy_), 30)
        pygame.draw.circle(self._screen, (55, 65, 75), (cx_, cy_), 30, 1)
        for angle, label in [(0,"E"),(math.pi/2,"S"),(math.pi,"W"),(3*math.pi/2,"N")]:
            lx = int(cx_ + 20 * math.cos(angle))
            ly = int(cy_ + 20 * math.sin(angle))
            l_surf = self._font_sm.render(label, True, _DIM_COL)
            self._screen.blit(l_surf, (lx - 5, ly - 7))
        # Heading needle
        nx = int(cx_ + 22 * math.cos(vehicle_yaw))
        ny = int(cy_ + 22 * math.sin(vehicle_yaw))
        pygame.draw.line(self._screen, _VEH_COL, (cx_, cy_), (nx, ny), 2)
        pygame.draw.circle(self._screen, _VEH_COL, (cx_, cy_), 3)

        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_speed(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def _carla_pos(tf) -> tuple[float, float]:
    return float(tf.location.x), float(tf.location.y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    # ── Connect to mock world ─────────────────────────────────────────
    client = carla.Client()
    client.set_timeout(5.0)
    world  = client.get_world()
    print(f"[AutoNav] Connected to {client.get_server_version()}")
    print(f"[AutoNav] Map: {world.get_map().name}")

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    actor_list = []
    sensors    = None
    dashboard  = None
    sim_view   = None

    try:
        # ── Spawn vehicle ─────────────────────────────────────────────
        bp_lib   = world.get_blueprint_library()
        car_bp   = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_pt = world.get_map().get_spawn_points()[0]
        vehicle  = world.spawn_actor(car_bp, spawn_pt)
        actor_list.append(vehicle)
        print(f"[AutoNav] Vehicle spawned (id={vehicle.id})")

        # ── Sensors ───────────────────────────────────────────────────
        sensors  = SensorManager(world, vehicle)

        # ── Pipeline ──────────────────────────────────────────────────
        K        = build_camera_matrix()
        vo       = VisualOdometry(K)
        # Centre the occupancy grid on the spawn point
        spawn_x  = float(spawn_pt.location.x)
        spawn_y  = float(spawn_pt.location.y)
        occ      = OccupancyGrid(origin_x=spawn_x, origin_y=spawn_y)
        ctrl     = PurePursuitController()
        dashboard = Dashboard()
        sim_view  = SimView(WORLD_SIZE)

        gt_positions:  list[tuple[float, float]] = []
        est_positions: list[tuple[float, float]] = []

        goal_world   = (float(args.goal_x), float(args.goal_y))
        print(f"[AutoNav] Goal: {goal_world}")

        planned_path: list[tuple[float, float]] | None = None
        tick_count   = 0
        prev_gt_pos: tuple[float, float] | None = None
        rmse         = 0.0

        print("[AutoNav] Running — press Q in either window to stop.")

        while True:
            world.tick()
            tick_count += 1

            # ── Ground truth ──────────────────────────────────────────
            tf      = vehicle.get_transform()
            gt_pos  = _carla_pos(tf)
            gt_yaw  = math.radians(tf.rotation.yaw)
            speed   = _get_speed(vehicle)

            # ── VO scale from GT displacement ─────────────────────────
            gt_scale = None
            if prev_gt_pos is not None:
                gt_scale = math.hypot(gt_pos[0] - prev_gt_pos[0],
                                      gt_pos[1] - prev_gt_pos[1])
            prev_gt_pos = gt_pos

            # ── Visual odometry ───────────────────────────────────────
            cam_frame = sensors.get_camera_frame()
            if cam_frame is not None:
                pose    = vo.update(cam_frame, gt_scale=gt_scale)
                est_pos = pose.position_2d
            else:
                est_pos = gt_pos

            gt_positions.append(gt_pos)
            est_positions.append(est_pos)

            # ── Occupancy grid ────────────────────────────────────────
            lidar_pts = sensors.get_lidar_points()
            occ.update(lidar_pts, gt_pos, gt_yaw)

            # ── Path planning ─────────────────────────────────────────
            if tick_count % 30 == 0 or planned_path is None:
                new_path = astar(occ, gt_pos, goal_world)
                if new_path is not None:
                    planned_path = new_path
                    ctrl.set_path(planned_path)
                    print(f"[AutoNav] Path planned: {len(planned_path)} wpts")

            # ── Control ───────────────────────────────────────────────
            if ctrl.has_path():
                sim_time = world.get_snapshot().timestamp.elapsed_seconds
                vc = ctrl.update(tf, speed, timestamp=sim_time)
                vehicle.apply_control(vc)
                if ctrl.path_complete(gt_pos, tolerance_m=4.0):
                    print("[AutoNav] Goal reached!")
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    break
            else:
                vehicle.apply_control(carla.VehicleControl(brake=1.0))

            # ── Drift stats ───────────────────────────────────────────
            if tick_count % 10 == 0 and len(gt_positions) > 5:
                d    = evaluate_drift(est_positions, gt_positions)
                rmse = d["rmse"]
                dashboard.rmse        = rmse
                dashboard.final_drift = d["final_drift"]

            # ── OpenCV dashboard ──────────────────────────────────────
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

            # ── Pygame sim view ───────────────────────────────────────
            keep_running = sim_view.render(
                vehicle_pos = gt_pos,
                vehicle_yaw = gt_yaw,
                path        = planned_path,
                gt_traj     = gt_positions,
                est_traj    = est_positions,
                goal        = goal_world,
                speed       = speed,
                rmse        = rmse,
            )
            if not keep_running:
                print("[AutoNav] Window closed by user.")
                break

            # Q in OpenCV window
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        # ── Final report ──────────────────────────────────────────────
        print("\n── VO Drift Report ───────────────────────────────────────")
        if len(gt_positions) > 1:
            d = evaluate_drift(est_positions, gt_positions)
            print(f"  Frames processed : {vo.frame_count}")
            print(f"  Frames skipped   : {vo.skipped_frames}")
            print(f"  RMSE             : {d['rmse']:.3f} m")
            print(f"  Max error        : {d['max_error']:.3f} m")
            print(f"  Final drift      : {d['final_drift']:.3f} m")
        else:
            print("  (not enough data)")

        os.makedirs("results", exist_ok=True)
        dashboard.save_trajectory_plot("results/trajectory_comparison.png")

    except KeyboardInterrupt:
        print("\n[AutoNav] Stopped by user.")
    except Exception:
        traceback.print_exc()
    finally:
        print("[AutoNav] Cleaning up ...")
        if sensors:
            sensors.destroy()
        for a in actor_list:
            if a.is_alive:
                a.destroy()
        if dashboard:
            dashboard.close()
        if sim_view:
            sim_view.close()
        print("[AutoNav] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoNav mock sim")
    parser.add_argument("--goal-x", type=float, default=80.0)
    parser.add_argument("--goal-y", type=float, default=120.0)
    args = parser.parse_args()
    run(args)
