"""Lightweight data-type stubs matching the carla Python API."""
from __future__ import annotations
import math
import numpy as np


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vector3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Location(Vector3D):
    pass


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw   = float(yaw)
        self.roll  = float(roll)


class Transform:
    def __init__(self, location: Location | None = None,
                 rotation: Rotation | None = None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self) -> Vector3D:
        yaw = math.radians(self.rotation.yaw)
        return Vector3D(math.cos(yaw), math.sin(yaw), 0.0)


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r, self.g, self.b, self.a = r, g, b, a


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0,
                 hand_brake=False, reverse=False, manual_gear_shift=False,
                 gear=1):
        self.throttle           = float(throttle)
        self.steer              = float(steer)
        self.brake              = float(brake)
        self.hand_brake         = hand_brake
        self.reverse            = reverse
        self.manual_gear_shift  = manual_gear_shift
        self.gear               = gear


class ActorBlueprint:
    def __init__(self, id_: str):
        self.id = id_
        self._attrs: dict[str, str] = {}

    def set_attribute(self, key: str, value: str):
        self._attrs[key] = value

    def get_attribute(self, key: str) -> str:
        return self._attrs.get(key, "")

    def __repr__(self):
        return f"ActorBlueprint({self.id})"


class BlueprintLibrary:
    def __init__(self, blueprints: list[ActorBlueprint]):
        self._bps = blueprints

    def find(self, id_: str) -> ActorBlueprint:
        for bp in self._bps:
            if bp.id == id_:
                return bp
        # return a generic one rather than crash
        return ActorBlueprint(id_)

    def filter(self, pattern: str) -> list[ActorBlueprint]:
        return [bp for bp in self._bps if pattern.replace("*", "") in bp.id] or [ActorBlueprint(pattern)]


class _RawBuffer:
    """Minimal stand-in for carla sensor data."""
    def __init__(self, raw: bytes, width: int, height: int):
        self.raw_data = raw
        self.width    = width
        self.height   = height


class Image(_RawBuffer):
    def __init__(self, array_bgra: "np.ndarray"):
        h, w = array_bgra.shape[:2]
        super().__init__(array_bgra.tobytes(), w, h)
        self.frame = 0


class LidarMeasurement:
    def __init__(self, points_xyzI: "np.ndarray"):
        """points_xyzI: (N,4) float32"""
        self.raw_data = points_xyzI.astype(np.float32).tobytes()
        self._n = len(points_xyzI)

    def __len__(self):
        return self._n
