"""
verify_setup.py — Minimal connection test.

Run this BEFORE main.py to confirm:
  1. CARLA server is reachable.
  2. You can spawn a vehicle.
  3. The RGB camera streams real data (one frame received).

Usage (with CARLA server already running):
    py -3.11 src/verify_setup.py

Expected output on success:
    [verify] Connected to CARLA <version>
    [verify] Map: /Game/Carla/Maps/Town01
    [verify] Vehicle spawned OK
    [verify] Camera attached — waiting for first frame ...
    [verify] Frame received: shape=(600, 800, 3)  dtype=uint8
    [verify] ✓ Setup looks good! You can run main.py now.
"""

import sys
import time
import numpy as np

try:
    import carla
except ImportError:
    print(
        "ERROR: 'carla' package not found.\n"
        "  Make sure you activated the venv and ran:\n"
        "    pip install -r requirements.txt\n"
        "  The carla package must match your installed CARLA version exactly.\n"
        "  Current latest: carla==0.9.16  (see README for details)"
    )
    sys.exit(1)

HOST    = "127.0.0.1"
PORT    = 2000
TIMEOUT = 10.0

def main():
    # ── Connect ──────────────────────────────────────────────────────
    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT)
        world  = client.get_world()
    except RuntimeError as exc:
        print(
            f"\nERROR: Cannot connect to CARLA at {HOST}:{PORT}.\n"
            "  → Make sure CarlaUE4.exe (or CarlaUE5.exe) is running first.\n"
            "  → This script is the CLIENT; CARLA is the SERVER.\n"
            f"  Detail: {exc}"
        )
        sys.exit(1)

    version = client.get_server_version()
    print(f"[verify] Connected to CARLA {version}")
    print(f"[verify] Map: {world.get_map().name}")

    actor_list = []
    try:
        # ── Spawn vehicle ─────────────────────────────────────────────
        bp_lib   = world.get_blueprint_library()
        car_bp   = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_pt = world.get_map().get_spawn_points()[0]
        vehicle  = world.spawn_actor(car_bp, spawn_pt)
        actor_list.append(vehicle)
        print("[verify] Vehicle spawned OK")

        # ── Attach camera ─────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov",          "90")

        received_frame = [None]   # mutable container for callback

        def on_image(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            received_frame[0] = arr.reshape((img.height, img.width, 4))[:, :, :3]

        tf  = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam = world.spawn_actor(cam_bp, tf, attach_to=vehicle)
        actor_list.append(cam)
        cam.listen(on_image)
        print("[verify] Camera attached — waiting for first frame ...")

        # Give CARLA a few ticks to deliver the first image
        deadline = time.time() + 5.0
        while time.time() < deadline and received_frame[0] is None:
            world.tick()
            time.sleep(0.05)

        if received_frame[0] is None:
            print("ERROR: No camera frame received within 5 s.\n"
                  "  → Check that the CARLA server is rendering (not paused).")
            sys.exit(1)

        frame = received_frame[0]
        print(f"[verify] Frame received: shape={frame.shape}  dtype={frame.dtype}")
        print("[verify] ✓ Setup looks good! You can run main.py now.")

    finally:
        for actor in actor_list:
            if actor.is_alive:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()


if __name__ == "__main__":
    main()
