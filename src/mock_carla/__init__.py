# mock_carla — drop-in stub for the carla Python package.
# Exposes the same classes/functions our pipeline uses so every
# src/ module imports cleanly without a real CARLA installation.
from .types import (
    Location, Rotation, Transform, Vector3D, Color,
    VehicleControl, ActorBlueprint, BlueprintLibrary,
    Image, LidarMeasurement,
)
from .world import World, Client, Map, Actor, Vehicle, Sensor
