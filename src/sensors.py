"""
sensors.py — Sensor setup for the AutoNav stack.

Attaches an RGB camera and a LiDAR sensor to a CARLA vehicle,
and exposes a clean interface for reading frames/point-clouds
into the rest of the pipeline.

Both sensors run asynchronously: CARLA calls the registered
callback each time a new frame/sweep arrives, and we store the
latest reading in a thread-safe slot that the main loop polls.
"""

import threading
import queue
import numpy as np
try:
    import carla
except ImportError:
    import mock_carla as carla  # type: ignore


# ---------------------------------------------------------------------------
# Configuration constants — tweak to taste
# ---------------------------------------------------------------------------

# Camera intrinsics (must stay consistent with visual_odometry.py)
CAM_WIDTH   = 800          # pixels
CAM_HEIGHT  = 600
CAM_FOV     = 90.0         # degrees (horizontal)

# LiDAR settings
LIDAR_CHANNELS      = 32   # vertical scan lines
LIDAR_RANGE         = 50.0 # metres
LIDAR_POINTS_PER_S  = 100_000
LIDAR_ROTATION_FREQ = 10   # Hz  ← must match CARLA's fixed_delta_seconds


# ---------------------------------------------------------------------------
# Sensor manager
# ---------------------------------------------------------------------------

class SensorManager:
    """
    Attaches sensors to *actor* (a carla.Vehicle) and exposes:
      .get_camera_frame()  -> np.ndarray (H, W, 3) BGR, or None
      .get_lidar_points()  -> np.ndarray (N, 4) [x,y,z,intensity], or None
      .destroy()           -> cleans up CARLA actors
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self._world   = world
        self._vehicle = vehicle

        # Latest data slots (None until first frame arrives)
        self._cam_frame: np.ndarray | None = None
        self._lidar_pts: np.ndarray | None = None
        self._cam_lock  = threading.Lock()
        self._lidar_lock = threading.Lock()

        self._sensors: list[carla.Actor] = []
        self._attach_camera()
        self._attach_lidar()

    # ------------------------------------------------------------------
    # Attachment helpers
    # ------------------------------------------------------------------

    def _attach_camera(self):
        bp_lib  = self._world.get_blueprint_library()
        cam_bp  = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_WIDTH))
        cam_bp.set_attribute("image_size_y", str(CAM_HEIGHT))
        cam_bp.set_attribute("fov",          str(CAM_FOV))

        # Mount slightly above the vehicle's centre of mass, facing forward
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam = self._world.spawn_actor(cam_bp, transform,
                                      attach_to=self._vehicle)
        cam.listen(self._on_camera_image)
        self._sensors.append(cam)
        print(f"[SensorManager] Camera attached (id={cam.id})")

    def _attach_lidar(self):
        bp_lib   = self._world.get_blueprint_library()
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels",           str(LIDAR_CHANNELS))
        lidar_bp.set_attribute("range",              str(LIDAR_RANGE))
        lidar_bp.set_attribute("points_per_second",  str(LIDAR_POINTS_PER_S))
        lidar_bp.set_attribute("rotation_frequency", str(LIDAR_ROTATION_FREQ))

        # Same mounting point as camera (good enough for prototyping)
        transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        lidar = self._world.spawn_actor(lidar_bp, transform,
                                        attach_to=self._vehicle)
        lidar.listen(self._on_lidar_sweep)
        self._sensors.append(lidar)
        print(f"[SensorManager] LiDAR attached  (id={lidar.id})")

    # ------------------------------------------------------------------
    # CARLA callbacks (called from CARLA's internal thread)
    # ------------------------------------------------------------------

    def _on_camera_image(self, image: carla.Image):
        """Convert raw BGRA bytes → BGR numpy array and stash it."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr   = array[:, :, :3]          # drop alpha channel
        with self._cam_lock:
            self._cam_frame = bgr.copy()  # copy so the buffer is ours

    def _on_lidar_sweep(self, point_cloud: carla.LidarMeasurement):
        """
        Convert raw LiDAR bytes into a (N, 4) float32 array.
        CARLA's ray-cast LiDAR encodes each point as [x, y, z, intensity].
        Note: CARLA uses a left-handed coordinate system (x forward, y right,
        z up) — we keep it as-is; mapping.py handles the 2-D projection.
        """
        data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        data = data.reshape((-1, 4))
        with self._lidar_lock:
            self._lidar_pts = data.copy()

    # ------------------------------------------------------------------
    # Public accessors — call from your main loop
    # ------------------------------------------------------------------

    def get_camera_frame(self) -> np.ndarray | None:
        """Return the most recent BGR camera frame, or None if not yet ready."""
        with self._cam_lock:
            return self._cam_frame

    def get_lidar_points(self) -> np.ndarray | None:
        """
        Return the most recent LiDAR point cloud as (N, 4) [x,y,z,intensity],
        or None if not yet ready.
        """
        with self._lidar_lock:
            return self._lidar_pts

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Stop and destroy all attached sensor actors."""
        for sensor in self._sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors.clear()
        print("[SensorManager] All sensors destroyed.")


# ---------------------------------------------------------------------------
# Camera intrinsic matrix (shared with visual_odometry.py)
# ---------------------------------------------------------------------------

def build_camera_matrix(width: int = CAM_WIDTH,
                        height: int = CAM_HEIGHT,
                        fov_deg: float = CAM_FOV) -> np.ndarray:
    """
    Return the 3×3 pinhole camera intrinsic matrix K for the given
    resolution and horizontal field-of-view.

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    fx = fy = (width/2) / tan(fov/2)   (square pixels assumed)
    cx = width/2,  cy = height/2
    """
    import math
    f  = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    cx = width  / 2.0
    cy = height / 2.0
    K  = np.array([[f,  0, cx],
                   [0,  f, cy],
                   [0,  0,  1]], dtype=np.float64)
    return K
