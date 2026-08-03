"""
visualization.py — AutoNav live dashboard

Single OpenCV window, dark-themed, three panels:

  ┌──────────────────────┬──────────────────────────────┐
  │  Camera feed  640×480│  Occupancy map  640×640      │
  │  Speed · RMSE HUD    │  Path · vehicle · LiDAR pts  │
  ├──────────────────────┴──────────────────────────────┤
  │            GT vs VO Trajectory  1280×280            │
  └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from mapping import OccupancyGrid, GRID_CELLS, GRID_RESOLUTION

# ── Panel sizes ────────────────────────────────────────────────────────
CAM_W, CAM_H       = 640, 480
MAP_W, MAP_H       = 640, 640
TRAJ_W, TRAJ_H     = CAM_W + MAP_W, 280

# ── Dark UI palette  (BGR) ─────────────────────────────────────────────
C_BG          = ( 18,  20,  24)
C_PANEL       = ( 28,  32,  38)
C_BORDER      = ( 55,  65,  75)
C_TEXT        = (220, 225, 230)
C_TEXT_DIM    = (110, 120, 130)
C_ACCENT      = ( 60, 180, 255)   # cyan
C_GREEN       = ( 60, 210,  80)
C_RED         = ( 55,  70, 230)
C_YELLOW      = ( 30, 210, 255)
C_PATH        = ( 40, 200, 140)   # teal path
C_LIDAR       = ( 20, 120, 255)   # orange LiDAR dots
C_FREE        = ( 55,  68,  60)   # visible dark teal for known-free roads
C_OCC         = ( 40,  80, 180)   # warm red-orange for obstacles
C_UNKNOWN     = ( 22,  26,  30)   # near-black for unseen

WINDOW_NAME = "AutoNav Dashboard"

# ── Font helpers ───────────────────────────────────────────────────────
_F   = cv2.FONT_HERSHEY_SIMPLEX
_FM  = cv2.FONT_HERSHEY_DUPLEX

def _text(img, txt, pos, scale=0.52, colour=C_TEXT, bold=1):
    cv2.putText(img, txt, pos, _F, scale, colour, bold, cv2.LINE_AA)

def _text2(img, txt, pos, scale=0.58, colour=C_TEXT, bold=1):
    cv2.putText(img, txt, pos, _FM, scale, colour, bold, cv2.LINE_AA)

def _panel_header(img, label: str, y: int = 22, x: int = 12):
    """Draw a small coloured accent bar + label."""
    cv2.rectangle(img, (x, y - 14), (x + 3, y + 4), C_ACCENT, -1)
    _text2(img, label, (x + 10, y), scale=0.52, colour=C_ACCENT)


class Dashboard:
    def __init__(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, CAM_W + MAP_W, max(CAM_H, MAP_H) + TRAJ_H)
        self._gt_traj:  list[tuple[float, float]] = []
        self._est_traj: list[tuple[float, float]] = []
        self.rmse        = 0.0
        self.final_drift = 0.0
        self._tick       = 0

    # ------------------------------------------------------------------
    def update(self, camera_bgr, occ_grid, vehicle_pos_gt, vehicle_pos_est,
               planned_path, vehicle_yaw_rad=0.0, speed_mps=0.0,
               lookahead_point=None) -> None:
        self._gt_traj.append(vehicle_pos_gt)
        self._est_traj.append(vehicle_pos_est)
        self._tick += 1

        cam_tile  = self._camera_panel(camera_bgr, speed_mps)
        map_tile  = self._map_panel(occ_grid, vehicle_pos_gt, vehicle_pos_est,
                                    planned_path, vehicle_yaw_rad, lookahead_point)
        traj_tile = self._trajectory_panel()

        # pad cam tile height to match map
        if cam_tile.shape[0] < MAP_H:
            pad = np.full((MAP_H - cam_tile.shape[0], CAM_W, 3),
                          C_BG[0], dtype=np.uint8)
            pad[:, :] = C_BG
            cam_tile = np.vstack([cam_tile, pad])

        top  = np.hstack([cam_tile, map_tile])
        full = np.vstack([top, traj_tile])

        # Outer border
        cv2.rectangle(full, (0, 0), (full.shape[1]-1, full.shape[0]-1),
                      C_BORDER, 1)

        cv2.imshow(WINDOW_NAME, full)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Camera panel
    # ------------------------------------------------------------------
    def _camera_panel(self, frame, speed_mps) -> np.ndarray:
        panel = np.full((CAM_H, CAM_W, 3), C_BG, dtype=np.uint8)

        if frame is not None:
            img = cv2.resize(frame, (CAM_W, CAM_H - 60))
            # subtle vignette
            h, w = img.shape[:2]
            vig = np.zeros((h, w), np.float32)
            cx, cy = w // 2, h // 2
            Y, X = np.ogrid[:h, :w]
            vig = 1.0 - np.clip(((X-cx)**2/(cx**2) + (Y-cy)**2/(cy**2)), 0, 1) * 0.45
            img = np.clip(img * vig[:,:,None], 0, 255).astype(np.uint8)
            panel[0:CAM_H-60, :] = img

        # ── Bottom HUD strip ──────────────────────────────────────────
        hud_y = CAM_H - 58
        cv2.rectangle(panel, (0, hud_y), (CAM_W, CAM_H), C_PANEL, -1)
        cv2.line(panel, (0, hud_y), (CAM_W, hud_y), C_BORDER, 1)

        # Speed — large
        kmh = speed_mps * 3.6
        _text2(panel, f"{kmh:5.1f}", (16, hud_y + 38), scale=1.0, colour=C_ACCENT)
        _text(panel,  "km/h",        (90, hud_y + 38), scale=0.45, colour=C_TEXT_DIM)

        # Separator
        cv2.line(panel, (130, hud_y + 8), (130, CAM_H - 8), C_BORDER, 1)

        # RMSE
        rmse_col = C_GREEN if self.rmse < 3.0 else (C_YELLOW if self.rmse < 8.0 else C_RED)
        _text(panel,  "VO DRIFT",          (142, hud_y + 20), scale=0.42, colour=C_TEXT_DIM)
        _text2(panel, f"{self.rmse:.2f} m",(142, hud_y + 44), scale=0.7,  colour=rmse_col)

        # Separator
        cv2.line(panel, (290, hud_y + 8), (290, CAM_H - 8), C_BORDER, 1)

        # Tick counter
        _text(panel, "TICK",              (302, hud_y + 20), scale=0.42, colour=C_TEXT_DIM)
        _text2(panel, f"{self._tick:5d}", (302, hud_y + 44), scale=0.7,  colour=C_TEXT)

        # Panel header
        _panel_header(panel, "FORWARD CAMERA")

        # Corner crosshair reticle
        mid_x, mid_y = CAM_W // 2, (CAM_H - 60) // 2
        for dx, dy in [(-18,0),(18,0),(0,-18),(0,18)]:
            cv2.line(panel, (mid_x + dx, mid_y + dy),
                     (mid_x + dx + (4 if dx else 0),
                      mid_y + dy + (4 if dy else 0)),
                     (80, 90, 100), 1)
        cv2.circle(panel, (mid_x, mid_y), 3, (80, 90, 100), 1)

        return panel

    # ------------------------------------------------------------------
    # Occupancy map panel  — LOCAL view centred on vehicle
    # ------------------------------------------------------------------
    def _map_panel(self, grid, pos_gt, pos_est, path,
                   yaw_rad, lookahead) -> np.ndarray:

        # ── Build a local crop: VIEW_M metres around the vehicle ──────
        VIEW_M   = 60.0          # metres visible each side
        CROP_PX  = MAP_W         # output pixels (square)
        m_per_px = (VIEW_M * 2) / CROP_PX

        vx, vy = pos_gt

        # Full probability grid
        prob = grid.get_probability()

        # Colour full grid
        full = np.full((GRID_CELLS, GRID_CELLS, 3), C_UNKNOWN, dtype=np.uint8)
        full[prob < 0.38] = C_FREE
        full[prob > 0.62] = C_OCC

        # Vehicle cell in full grid
        vr, vc = grid.world_to_cell(vx, vy)

        # How many full-grid cells correspond to VIEW_M?
        cells_per_view = int(VIEW_M / GRID_RESOLUTION)

        r0 = max(vr - cells_per_view, 0)
        r1 = min(vr + cells_per_view, GRID_CELLS)
        c0 = max(vc - cells_per_view, 0)
        c1 = min(vc + cells_per_view, GRID_CELLS)

        crop = full[r0:r1, c0:c1]

        # Resize crop to CROP_PX × CROP_PX
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            img = cv2.resize(crop, (CROP_PX, CROP_PX), interpolation=cv2.INTER_NEAREST)
        else:
            img = np.full((CROP_PX, CROP_PX, 3), C_UNKNOWN, dtype=np.uint8)

        # Scale factor: full-grid cell → crop pixel
        scale_r = CROP_PX / max(r1 - r0, 1)
        scale_c = CROP_PX / max(c1 - c0, 1)

        def w2crop(wx, wy):
            """World → pixel in the cropped/scaled view."""
            gr, gc = grid.world_to_cell(wx, wy)
            px_c = int((gc - c0) * scale_c)
            px_r = int((gr - r0) * scale_r)
            return (np.clip(px_c, 0, CROP_PX-1),
                    np.clip(px_r, 0, CROP_PX-1))

        # ── Planned path ──────────────────────────────────────────────
        if path and len(path) > 1:
            pts = np.array([w2crop(*p) for p in path], dtype=np.int32)
            # Glow
            cv2.polylines(img, [pts], False, (30, 140, 100), 3, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, C_PATH, 2, cv2.LINE_AA)
            # Dots every 10 waypoints
            for i in range(0, len(pts), 10):
                cv2.circle(img, tuple(pts[i]), 3, C_PATH, -1)

        # ── Goal star (if in view) ─────────────────────────────────────
        if path:
            gp_ = w2crop(*path[-1])
            cv2.drawMarker(img, gp_, C_YELLOW, cv2.MARKER_STAR, 14, 2)

        # ── Lookahead ─────────────────────────────────────────────────
        if lookahead:
            lp = w2crop(*lookahead)
            cv2.circle(img, lp, 6, C_YELLOW, -1)
            cv2.circle(img, lp, 9, C_YELLOW,  1)

        # ── VO estimated pos ──────────────────────────────────────────
        ep = w2crop(*pos_est)
        cv2.circle(img, ep, 6, (30, 120, 255), -1)
        cv2.circle(img, ep, 10, (30, 120, 255), 1)

        # ── Ground-truth vehicle ──────────────────────────────────────
        gp = w2crop(*pos_gt)
        # Car body: small rotated rect
        body_len, body_w = 14, 8
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        car_pts = np.array([
            (int(gp[0] + cos_y*body_len//2 - sin_y*body_w//2),
             int(gp[1] + sin_y*body_len//2 + cos_y*body_w//2)),
            (int(gp[0] - cos_y*body_len//2 - sin_y*body_w//2),
             int(gp[1] - sin_y*body_len//2 + cos_y*body_w//2)),
            (int(gp[0] - cos_y*body_len//2 + sin_y*body_w//2),
             int(gp[1] - sin_y*body_len//2 - cos_y*body_w//2)),
            (int(gp[0] + cos_y*body_len//2 + sin_y*body_w//2),
             int(gp[1] + sin_y*body_len//2 - cos_y*body_w//2)),
        ], dtype=np.int32)
        cv2.fillPoly(img, [car_pts], C_GREEN)
        cv2.polylines(img, [car_pts], True, (200,255,200), 1)
        # Heading arrow
        ax = int(gp[0] + 18 * cos_y)
        ay = int(gp[1] + 18 * sin_y)
        cv2.arrowedLine(img, gp, (ax, ay), (255,255,255), 2,
                        tipLength=0.4, line_type=cv2.LINE_AA)

        # ── North indicator ───────────────────────────────────────────
        cv2.arrowedLine(img, (CROP_PX-22, CROP_PX-44),
                        (CROP_PX-22, CROP_PX-22), (180,180,180), 1, tipLength=0.3)
        _text(img, "N", (CROP_PX-27, CROP_PX-46), scale=0.35, colour=(180,180,180))

        # ── View-range scale bar ──────────────────────────────────────
        bar_m = 20   # metres
        bar_px = int(bar_m / (VIEW_M * 2) * CROP_PX)
        cv2.rectangle(img, (12, CROP_PX-18), (12+bar_px, CROP_PX-14), (160,160,160), -1)
        _text(img, f"{bar_m}m", (14+bar_px, CROP_PX-10), scale=0.35, colour=C_TEXT_DIM)

        tile = img.copy()

        # ── Header ────────────────────────────────────────────────────
        cv2.rectangle(tile, (0, 0), (MAP_W, 30), C_PANEL, -1)
        cv2.line(tile, (0, 30), (MAP_W, 30), C_BORDER, 1)
        _panel_header(tile, "OCCUPANCY MAP  +  PLANNED PATH")
        _text(tile, f"view {int(VIEW_M*2)}m x {int(VIEW_M*2)}m",
              (MAP_W-120, 22), scale=0.38, colour=C_TEXT_DIM)

        # ── Legend ────────────────────────────────────────────────────
        cv2.rectangle(tile, (0, MAP_H-24), (MAP_W, MAP_H), C_PANEL, -1)
        cv2.line(tile, (0, MAP_H-24), (MAP_W, MAP_H-24), C_BORDER, 1)

        def _leg(x, col, label):
            cv2.rectangle(tile, (x, MAP_H-16), (x+10, MAP_H-6), col, -1)
            _text(tile, label, (x+14, MAP_H-6), scale=0.37, colour=C_TEXT_DIM)

        _leg(10,  C_FREE,         "free")
        _leg(60,  C_OCC,          "obstacle")
        _leg(152, C_GREEN,        "vehicle")
        _leg(228, (30,120,255),   "VO est")
        _leg(296, C_PATH,         "path")
        _leg(348, C_YELLOW,       "lookahead")

        cv2.line(tile, (0, 0), (0, MAP_H), C_BORDER, 2)
        return tile

    # ------------------------------------------------------------------
    # Trajectory panel
    # ------------------------------------------------------------------
    def _trajectory_panel(self) -> np.ndarray:
        tile = np.full((TRAJ_H, TRAJ_W, 3), C_PANEL, dtype=np.uint8)

        # Header
        cv2.rectangle(tile, (0, 0), (TRAJ_W, 28), C_BG, -1)
        cv2.line(tile, (0, 28), (TRAJ_W, 28), C_BORDER, 1)
        _panel_header(tile, "TRAJECTORY  |  Ground Truth vs VO Estimate")

        # Legend — use plain ASCII boxes instead of Unicode
        cv2.rectangle(tile, (TRAJ_W-330, 8), (TRAJ_W-318, 20), C_GREEN, -1)
        _text(tile, "Ground truth", (TRAJ_W-314, 20), scale=0.42, colour=C_GREEN)
        cv2.rectangle(tile, (TRAJ_W-170, 8), (TRAJ_W-158, 20), C_RED, -1)
        _text(tile, "VO estimate",  (TRAJ_W-154, 20), scale=0.42, colour=C_RED)

        margin = 30
        plot_w = TRAJ_W  - 2 * margin
        plot_h = TRAJ_H  - 2 * margin - 10
        plot_x0, plot_y0 = margin, 35
        plot_x1 = plot_x0 + plot_w
        plot_y1 = plot_y0 + plot_h

        # Plot border
        cv2.rectangle(tile, (plot_x0, plot_y0), (plot_x1, plot_y1), C_BORDER, 1)

        all_pos = self._gt_traj + self._est_traj
        if len(all_pos) < 2:
            _text(tile, "Collecting data...", (plot_x0 + 20, plot_y0 + plot_h // 2),
                  scale=0.5, colour=C_TEXT_DIM)
            return tile

        xs = [p[0] for p in all_pos]
        ys = [p[1] for p in all_pos]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        # keep aspect ratio — square extent
        span = max(span_x, span_y) * 1.05
        cx_w = (min_x + max_x) / 2
        cy_w = (min_y + max_y) / 2

        def to_px(wx, wy):
            px = int(plot_x0 + (wx - (cx_w - span/2)) / span * plot_w)
            py = int(plot_y1 - (wy - (cy_w - span/2)) / span * plot_h)
            return (np.clip(px, plot_x0, plot_x1),
                    np.clip(py, plot_y0, plot_y1))

        # Grid lines
        for gi in range(1, 5):
            gx = plot_x0 + plot_w * gi // 4
            gy = plot_y0 + plot_h * gi // 4
            cv2.line(tile, (gx, plot_y0), (gx, plot_y1), C_BORDER, 1)
            cv2.line(tile, (plot_x0, gy), (plot_x1, gy), C_BORDER, 1)

        def draw_traj(traj, colour, thickness=2):
            if len(traj) < 2:
                return
            pts = np.array([to_px(*p) for p in traj], dtype=np.int32)
            cv2.polylines(tile, [pts], False, colour, thickness, cv2.LINE_AA)
            # Start dot
            cv2.circle(tile, tuple(pts[0]),  4, colour, -1)
            # End dot (larger)
            cv2.circle(tile, tuple(pts[-1]), 6, colour, -1)

        draw_traj(self._gt_traj,  C_GREEN, 2)
        draw_traj(self._est_traj, C_RED,   2)

        # Start / end labels
        if self._gt_traj:
            sp = to_px(*self._gt_traj[0])
            cv2.circle(tile, sp, 5, (200,200,200), -1)
            _text(tile, "START", (sp[0]+6, sp[1]-4), scale=0.36, colour=C_TEXT_DIM)
        if self._gt_traj:
            ep = to_px(*self._gt_traj[-1])
            _text(tile, "NOW", (ep[0]+6, ep[1]-4), scale=0.36, colour=C_TEXT_DIM)

        # Stats bar
        stats_y = TRAJ_H - 12
        rmse_col = C_GREEN if self.rmse < 3 else (C_YELLOW if self.rmse < 8 else C_RED)
        _text(tile, f"RMSE: {self.rmse:.2f} m",
              (margin, stats_y), scale=0.44, colour=rmse_col)
        _text(tile, f"Final drift: {self.final_drift:.2f} m",
              (margin + 180, stats_y), scale=0.44, colour=C_TEXT_DIM)
        _text(tile, f"Samples: {len(self._gt_traj)}",
              (margin + 400, stats_y), scale=0.44, colour=C_TEXT_DIM)

        return tile

    # ------------------------------------------------------------------
    def save_trajectory_plot(self, path: str):
        tile = self._trajectory_panel()
        cv2.imwrite(path, tile)
        print(f"[Dashboard] Trajectory plot saved → {path}")

    def close(self):
        cv2.destroyAllWindows()


def _draw_diamond(img, cx, cy, half, colour):
    pts = np.array([[cx, cy-half],[cx+half, cy],
                    [cx, cy+half],[cx-half, cy]], dtype=np.int32)
    cv2.fillPoly(img, [pts], colour)
