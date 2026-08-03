"""
mock_carla/world.py  — synthetic environment for AutoNav

Camera renderer: proper raycaster with sky gradient, atmospheric depth
haze, textured building facades, road surface with lane markings,
pavement kerbs and grass verges.

LiDAR: vectorised numpy ray-march, returns real hit points.
Vehicle: kinematic bicycle model.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from .types import (
    Location, Rotation, Transform, Vector3D, VehicleControl,
    ActorBlueprint, BlueprintLibrary, Image, LidarMeasurement,
)

# ---------------------------------------------------------------------------
# World geometry
# ---------------------------------------------------------------------------
ROAD_WIDTH  = 8.0
BLOCK_SIZE  = 50.0
N_BLOCKS    = 5
WORLD_SIZE  = N_BLOCKS * BLOCK_SIZE   # 250 m

def _build_obstacles():
    obs = []
    setback = ROAD_WIDTH / 2 + 3.0
    inner   = max(BLOCK_SIZE / 2 - setback, 5.0)
    for bx in range(N_BLOCKS):
        for by in range(N_BLOCKS):
            cx = bx * BLOCK_SIZE + BLOCK_SIZE / 2
            cy = by * BLOCK_SIZE + BLOCK_SIZE / 2
            obs.append((cx, cy, inner, inner))
    return obs

OBSTACLES = _build_obstacles()

_OBS_CX = np.array([o[0] for o in OBSTACLES], dtype=np.float64)
_OBS_CY = np.array([o[1] for o in OBSTACLES], dtype=np.float64)
_OBS_HW = np.array([o[2] for o in OBSTACLES], dtype=np.float64)
_OBS_HH = np.array([o[3] for o in OBSTACLES], dtype=np.float64)

# Building "colour" — each block gets a hue slot so they look distinct
_BUILDING_PALETTES = [
    (180, 160, 140),   # warm sandstone
    (130, 145, 160),   # cool concrete
    (160, 140, 120),   # terracotta
    (145, 155, 145),   # sage
    (170, 155, 130),   # beige
]

def _obs_colour(obs_idx: int, dist: float, face_normal_dot: float) -> tuple:
    """Return a shaded BGR colour for a building face."""
    base = _BUILDING_PALETTES[obs_idx % len(_BUILDING_PALETTES)]
    # Directional shading (darker on side faces)
    shade = max(0.35, min(1.0, 0.55 + 0.45 * abs(face_normal_dot)))
    # Distance fog: blend toward (180,190,200) at long range
    fog = min(1.0, dist / 55.0)
    fog_col = (190, 195, 200)
    r = int(base[0] * shade * (1 - fog) + fog_col[0] * fog)
    g = int(base[1] * shade * (1 - fog) + fog_col[1] * fog)
    b = int(base[2] * shade * (1 - fog) + fog_col[2] * fog)
    return (b, g, r)   # OpenCV is BGR

# ---------------------------------------------------------------------------
# Camera constants
# ---------------------------------------------------------------------------
CAM_W   = 800
CAM_H   = 600
CAM_FOV = 90.0
_RENDER_W = 400   # render at half-res, upscale to CAM_W×CAM_H
_RENDER_H = 300
_HORIZON_Y = _RENDER_H // 2 + 10
_PROJ_DIST = (_RENDER_W / 2.0) / math.tan(math.radians(CAM_FOV / 2.0))
_COL_OFFSET = np.arange(_RENDER_W, dtype=np.float64) - _RENDER_W / 2.0
_COL_ANGLES = np.arctan2(_COL_OFFSET, _PROJ_DIST)


def _generate_camera_image(vehicle: "Vehicle") -> Image:
    img = np.zeros((_RENDER_H, _RENDER_W, 3), dtype=np.uint8)
    vx, vy, vyaw = vehicle._x, vehicle._y, vehicle._yaw
    cos_yaw, sin_yaw = math.cos(vyaw), math.sin(vyaw)

    # ── 1. Sky gradient ───────────────────────────────────────────────
    sky_rows = np.arange(_HORIZON_Y, dtype=np.float32)
    t = sky_rows / max(_HORIZON_Y - 1, 1)
    sky_b = (160 + t * 80).astype(np.uint8)
    sky_g = (80  + t * 140).astype(np.uint8)
    sky_r = (40  + t * 140).astype(np.uint8)
    img[:_HORIZON_Y, :, 0] = sky_b[:, None]
    img[:_HORIZON_Y, :, 1] = sky_g[:, None]
    img[:_HORIZON_Y, :, 2] = sky_r[:, None]

    # Sun glow
    sun_x, sun_y = int(_RENDER_W * 0.68), int(_HORIZON_Y * 0.50)
    for radius, alpha in [(28, 0.07), (16, 0.13), (8, 0.28), (4, 0.60)]:
        ov = img.copy()
        cv2.circle(ov, (sun_x, sun_y), radius, (200, 230, 255), -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

    # Horizon haze
    hz = img[max(0,_HORIZON_Y-5):_HORIZON_Y].astype(np.float32)
    img[max(0,_HORIZON_Y-5):_HORIZON_Y] = np.clip(
        hz * 0.35 + np.array([220, 215, 200]) * 0.65, 0, 255).astype(np.uint8)

    # ── 2. Floor — proper perspective floor-casting ────────────────────
    # For every floor pixel (row, col) compute the world point it maps to.
    # Standard floor-casting formula from raycasting tutorials:
    #   rowDistance = posZ / (row - screenHeight/2)   where posZ = 0.5 * screenHeight
    # We use the camera plane (left/right vectors) for correctness.

    # Camera plane vectors (perpendicular to view direction, scaled by tan(fov/2))
    plane_x = -sin_yaw   # camera plane x (perpendicular to look dir)
    plane_y =  cos_yaw   # camera plane y

    half_h = _RENDER_H / 2.0
    rows = np.arange(_HORIZON_Y, _RENDER_H)         # floor rows
    row_dist = (half_h / np.maximum(rows - half_h, 0.01))  # distance to floor at this row

    # Column fraction in [-1, 1]
    col_frac = (np.arange(_RENDER_W) / _RENDER_W * 2.0 - 1.0)  # (W,)

    # For each (row, col): world X/Y
    # floor_x[r,c] = vx + row_dist[r] * (cos_yaw + plane_x * col_frac[c])
    rd   = row_dist[:, None]       # (H_floor, 1)
    cf   = col_frac[None, :]       # (1, W)
    wx2d = vx + rd * (cos_yaw + plane_x * cf)   # (H_floor, W)
    wy2d = vy + rd * (sin_yaw + plane_y * cf)   # (H_floor, W)

    bx2d = wx2d % BLOCK_SIZE
    by2d = wy2d % BLOCK_SIZE
    hw   = ROAD_WIDTH / 2

    on_rx   = (bx2d < hw) | (bx2d > BLOCK_SIZE - hw)
    on_ry   = (by2d < hw) | (by2d > BLOCK_SIZE - hw)
    on_road = on_rx | on_ry

    kerb = (
        ((bx2d >= hw)                   & (bx2d < hw + 0.9)) |
        ((bx2d > BLOCK_SIZE - hw - 0.9) & (bx2d <= BLOCK_SIZE - hw)) |
        ((by2d >= hw)                   & (by2d < hw + 0.9)) |
        ((by2d > BLOCK_SIZE - hw - 0.9) & (by2d <= BLOCK_SIZE - hw))
    )
    pavement = ~on_road & ~kerb & (
        ((bx2d >= hw)                   & (bx2d < hw + 3.0)) |
        ((bx2d > BLOCK_SIZE - hw - 3.0) & (bx2d <= BLOCK_SIZE - hw)) |
        ((by2d >= hw)                   & (by2d < hw + 3.0)) |
        ((by2d > BLOCK_SIZE - hw - 3.0) & (by2d <= BLOCK_SIZE - hw))
    )

    # Build RGB colour array (H_floor, W, 3)
    col_arr = np.zeros((_RENDER_H - _HORIZON_Y, _RENDER_W, 3), dtype=np.float32)

    # Grass
    g_n = ((wx2d * 5.1 + wy2d * 3.7).astype(int) % 7) * 3
    col_arr[..., 0] = 45
    col_arr[..., 1] = 88 + g_n
    col_arr[..., 2] = 42

    # Road asphalt
    r_n = ((wx2d * 3.7 + wy2d * 2.3).astype(int) % 5) * 3
    road_b = (50 + r_n).astype(np.float32)
    col_arr[on_road, 0] = road_b[on_road]
    col_arr[on_road, 1] = road_b[on_road]
    col_arr[on_road, 2] = road_b[on_road] + 3

    # Lane markings — dashed white centre lines
    mark_x = (np.abs(bx2d - BLOCK_SIZE/2) < 0.20) & on_rx & ((wy2d * 1.5).astype(int) % 5 < 3)
    mark_y = (np.abs(by2d - BLOCK_SIZE/2) < 0.20) & on_ry & ((wx2d * 1.5).astype(int) % 5 < 3)
    mark   = mark_x | mark_y
    col_arr[mark] = [200, 200, 200]

    # Pavement
    col_arr[pavement] = [148, 148, 143]

    # Kerb
    col_arr[kerb] = [190, 185, 175]

    # Distance fog
    dist_f = np.clip(rd.repeat(_RENDER_W, axis=1) / 50.0, 0, 1)[..., None]
    fog_c  = np.array([[[195, 195, 190]]], dtype=np.float32)
    col_arr = col_arr * (1 - dist_f) + fog_c * dist_f

    # Write BGR
    img[_HORIZON_Y:, :, 0] = col_arr[..., 2].astype(np.uint8)
    img[_HORIZON_Y:, :, 1] = col_arr[..., 1].astype(np.uint8)
    img[_HORIZON_Y:, :, 2] = col_arr[..., 0].astype(np.uint8)

    # ── 3. Building walls — per-column raycaster ──────────────────────
    ray_angles = vyaw + _COL_ANGLES
    cos_a = np.cos(ray_angles)
    sin_a = np.sin(ray_angles)

    step    = 0.5
    max_d   = 70.0
    n_steps = int(max_d / step)
    dists   = np.full(_RENDER_W, max_d)
    obs_ids = np.full(_RENDER_W, -1, dtype=int)
    found   = np.zeros(_RENDER_W, dtype=bool)
    active  = np.ones(_RENDER_W, dtype=bool)

    for s in range(1, n_steps + 1):
        d   = s * step
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        wx_ = vx + d * cos_a[idx]
        wy_ = vy + d * sin_a[idx]
        dx_obs = np.abs(wx_[None, :] - _OBS_CX[:, None])
        dy_obs = np.abs(wy_[None, :] - _OBS_CY[:, None])
        hit_mat = (dx_obs < _OBS_HW[:, None]) & (dy_obs < _OBS_HH[:, None])
        ray_hit = hit_mat.any(axis=0)
        if ray_hit.any():
            hg = idx[ray_hit]
            dists[hg]   = d
            obs_ids[hg] = hit_mat[:, ray_hit].argmax(axis=0)
            found[hg]   = True
            active[hg]  = False

    hit_cols = np.where(found)[0]
    if len(hit_cols):
        corr    = dists[hit_cols] * np.abs(np.cos(_COL_ANGLES[hit_cols]))
        col_h_v = np.clip((_PROJ_DIST / np.maximum(corr, 0.1)).astype(int), 0, _RENDER_H)
        tops_v  = np.maximum(_HORIZON_Y - col_h_v // 2, 0)
        bots_v  = np.minimum(_HORIZON_Y + col_h_v // 2, _RENDER_H)

        hx  = vx + dists[hit_cols] * cos_a[hit_cols]
        hy  = vy + dists[hit_cols] * sin_a[hit_cols]
        oi  = obs_ids[hit_cols]
        dxf = hx - _OBS_CX[oi];  dyf = hy - _OBS_CY[oi]
        fdt = np.where(np.abs(dxf) > np.abs(dyf),
                       np.abs(cos_a[hit_cols]), np.abs(sin_a[hit_cols]))

        pal       = np.array(_BUILDING_PALETTES, dtype=np.float32)
        base_rgb  = pal[oi % len(pal)]
        shade     = np.clip(0.55 + 0.45 * fdt, 0.35, 1.0)[:, None]
        fog_w     = np.clip(dists[hit_cols] / 55.0, 0, 1)[:, None]
        fog_c2    = np.array([[190, 195, 200]], dtype=np.float32)
        base_sh   = (base_rgb * shade * (1-fog_w) + fog_c2 * fog_w)

        for i, px in enumerate(hit_cols):
            top, bot = int(tops_v[i]), int(bots_v[i])
            if bot <= top: continue
            hw2 = bot - top
            bc  = base_sh[i]   # RGB float

            t_arr       = np.linspace(0, 1, hw2)
            wr          = (t_arr * 6).astype(int)
            wc          = int((px / (dists[hit_cols[i]] + 1)) * 4 + oi[i]) % 4
            in_win      = (wr % 2 == 1) & (wc % 2 == 0)
            in_frm      = (wr % 2 == 1) & (wc % 2 == 1)
            lit         = ((oi[i]*7 + wr*3 + wc*5) % 4 != 0)
            mortar      = ((t_arr * hw2 / 5).astype(int) % 3 == 0)

            strip = np.zeros((hw2, 3), dtype=np.uint8)
            strip[:, 0] = int(bc[2])   # B
            strip[:, 1] = int(bc[1])   # G
            strip[:, 2] = int(bc[0])   # R
            strip[mortar, :] = np.clip(strip[mortar].astype(int) - 15, 0, 255)
            strip[in_frm, :] = np.clip(strip[in_frm].astype(int)  - 20, 0, 255)
            strip[in_win &  lit] = [40, 180, 220]
            strip[in_win & ~lit] = [20,  25,  35]
            img[top:bot, px] = strip
            img[top:top+max(1,hw2//22), px] = (30, 35, 40)

    # ── 4. Chrome strip at bottom ─────────────────────────────────────
    bar_h = _RENDER_H // 8
    bar_y = _RENDER_H - bar_h
    chrome = img[bar_y:].astype(np.float32)
    img[bar_y:] = np.clip(chrome*0.25 + np.array([25,28,32])*0.75, 0, 255).astype(np.uint8)
    cv2.line(img, (0, bar_y), (_RENDER_W, bar_y), (60, 65, 70), 1)

    img = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return Image(bgra)


# ---------------------------------------------------------------------------
# LiDAR  (unchanged — already vectorised and correct)
# ---------------------------------------------------------------------------
def _generate_lidar(vehicle: "Vehicle",
                    n_channels: int = 16,
                    max_range: float = 50.0,
                    n_horiz: int = 128) -> LidarMeasurement:
    vx, vy, vyaw = vehicle._x, vehicle._y, vehicle._yaw
    h_angs  = np.linspace(0.0, 2 * math.pi, n_horiz, endpoint=False)
    v_angs  = np.linspace(math.radians(-10), math.radians(3), n_channels)
    H, V    = np.meshgrid(h_angs, v_angs)
    world_h = vyaw + H
    cos_v   = np.cos(V);  sin_v = np.sin(V)
    dx_all  = (np.cos(world_h) * cos_v).ravel()
    dy_all  = (np.sin(world_h) * cos_v).ravel()
    dz_all  = sin_v.ravel()
    n_rays  = len(dx_all)
    dists   = np.full(n_rays, max_range)
    found   = np.zeros(n_rays, dtype=bool)
    active  = np.ones(n_rays, dtype=bool)
    step    = 1.0
    n_steps = int(max_range / step)
    for s in range(1, n_steps + 1):
        d   = s * step
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        wx = vx + d * dx_all[idx]
        wy = vy + d * dy_all[idx]
        dx_obs = np.abs(wx[None, :] - _OBS_CX[:, None])
        dy_obs = np.abs(wy[None, :] - _OBS_CY[:, None])
        hit    = ((dx_obs < _OBS_HW[:, None]) & (dy_obs < _OBS_HH[:, None])).any(axis=0)
        hi              = idx[hit]
        dists[hi]       = d
        found[hi]       = True
        active[hi]      = False
    if not found.any():
        return LidarMeasurement(np.array([[max_range, 0, 0, 0]], dtype=np.float32))
    d_h    = dists[found]
    dxh    = dx_all[found];  dyh = dy_all[found];  dzh = dz_all[found]
    hit_wx = d_h * dxh;      hit_wy = d_h * dyh
    cy_    = math.cos(-vyaw); sy_ = math.sin(-vyaw)
    sx     = hit_wx * cy_ - hit_wy * sy_
    sy_v   = hit_wx * sy_ + hit_wy * cy_
    sz     = d_h * dzh
    inten  = np.maximum(0.0, 1.0 - d_h / max_range)
    return LidarMeasurement(np.column_stack([sx, sy_v, sz, inten]).astype(np.float32))


# ---------------------------------------------------------------------------
# Stubs (unchanged)
# ---------------------------------------------------------------------------
class _Timestamp:
    def __init__(self, t): self.elapsed_seconds = t
class _Snapshot:
    def __init__(self, t): self.timestamp = _Timestamp(t)

class Actor:
    _next_id = 1
    def __init__(self):
        self.id = Actor._next_id; Actor._next_id += 1
        self.type_id = "actor"; self._alive = True
    @property
    def is_alive(self): return self._alive
    def destroy(self): self._alive = False

class Sensor(Actor):
    def __init__(self, bp, transform, parent):
        super().__init__()
        self.type_id = bp.id; self._bp = bp
        self._tf = transform; self._parent = parent
        self._callback = None; self._stopped = False
    def listen(self, cb): self._callback = cb
    def stop(self): self._stopped = True
    def _fire(self, data):
        if self._callback and not self._stopped: self._callback(data)

class Vehicle(Actor):
    WHEELBASE = 2.9; MAX_STEER = 0.6
    def __init__(self, transform):
        super().__init__()
        self.type_id = "vehicle.tesla.model3"
        self._x = float(transform.location.x)
        self._y = float(transform.location.y)
        self._yaw = math.radians(transform.rotation.yaw)
        self._v = 0.0; self._ctrl = VehicleControl()
    def apply_control(self, ctrl): self._ctrl = ctrl
    def get_transform(self):
        return Transform(Location(self._x, self._y, 0.0),
                         Rotation(yaw=math.degrees(self._yaw)))
    def get_velocity(self):
        return Vector3D(self._v * math.cos(self._yaw),
                        self._v * math.sin(self._yaw), 0.0)
    def tick(self, dt):
        c = self._ctrl
        accel  = c.throttle * 4.0 - c.brake * 8.0 - 0.3 * self._v
        self._v = float(np.clip(self._v + accel * dt, 0.0, 15.0))
        delta   = c.steer * self.MAX_STEER
        yaw_rate = (self._v * math.tan(delta) / self.WHEELBASE
                    if abs(self._v) > 0.01 else 0.0)
        self._yaw += yaw_rate * dt
        self._x = float(np.clip(self._x + self._v*math.cos(self._yaw)*dt, 1.0, WORLD_SIZE-1.0))
        self._y = float(np.clip(self._y + self._v*math.sin(self._yaw)*dt, 1.0, WORLD_SIZE-1.0))

class Map:
    def __init__(self): self.name = "/MockCarla/Maps/MockTown"
    def get_spawn_points(self):
        # Spawn heading northeast so buildings are visible immediately
        return [Transform(Location(0.0, 0.0, 0.0), Rotation(yaw=35.0))]

class World:
    def __init__(self):
        self._map = Map(); self._actors = []; self._sensors = []
        self._vehicle = None; self._t = 0.0; self._dt = 0.05
    def get_map(self): return self._map
    def get_settings(self): return _Settings()
    def apply_settings(self, s):
        if s.fixed_delta_seconds: self._dt = s.fixed_delta_seconds
    def get_blueprint_library(self):
        return BlueprintLibrary([ActorBlueprint("vehicle.tesla.model3"),
                                 ActorBlueprint("sensor.camera.rgb"),
                                 ActorBlueprint("sensor.lidar.ray_cast")])
    def spawn_actor(self, bp, transform, attach_to=None):
        if "vehicle" in bp.id:
            v = Vehicle(transform); self._vehicle = v
            self._actors.append(v); return v
        elif "camera" in bp.id or "lidar" in bp.id:
            s = Sensor(bp, transform, attach_to)
            self._sensors.append(s); self._actors.append(s); return s
        raise ValueError(f"Unknown blueprint: {bp.id}")
    def tick(self):
        self._t += self._dt
        if self._vehicle:
            self._vehicle.tick(self._dt)
            for sensor in self._sensors:
                if not sensor._stopped and sensor._callback:
                    if "camera" in sensor.type_id:
                        sensor._fire(_generate_camera_image(self._vehicle))
                    elif "lidar" in sensor.type_id:
                        sensor._fire(_generate_lidar(self._vehicle))
        return _Snapshot(self._t)
    def get_snapshot(self): return _Snapshot(self._t)

class _Settings:
    synchronous_mode = False; fixed_delta_seconds = None

class Client:
    def __init__(self, host="127.0.0.1", port=2000):
        self._world = World()
    def set_timeout(self, s): pass
    def get_server_version(self): return "0.9.16-mock"
    def get_world(self): return self._world
    def load_world(self, name): return self._world
