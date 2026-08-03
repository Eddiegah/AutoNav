# AutoNav — Perception-to-Path-Planning Autonomous Navigation in CARLA

A working autonomous navigation stack for a simulated vehicle, built on the
[CARLA](https://carla.org/) open-source simulator.

---

## Honest Scope Statement

**What AutoNav actually implements:**
- **Visual odometry (VO)** — feature-based frame-to-frame motion estimation
- **Occupancy-grid mapping** — real-time 2-D obstacle map from LiDAR
- **A\* path planning** — optimal grid-based path finding with obstacle inflation
- **Pure Pursuit control** — geometric path-following controller
- **Live dashboard** — camera feed, map, planned path, and drift comparison

**What this is NOT:**

AutoNav does **not** implement full SLAM (Simultaneous Localisation and
Mapping).  Full SLAM adds *loop closure* — the ability to recognise a
previously-visited location and use that constraint to correct accumulated
drift across the entire trajectory.  Loop closure requires a place-recognition
module, a global pose graph, and a non-linear optimiser (e.g. g2o, GTSAM).
That is genuinely research-grade complexity and is out of scope here.

The README describes this project as what it is: visual odometry +
occupancy-grid mapping + path planning.  The drift that accumulates in the VO
trajectory is real, expected, and honestly reported — not hidden.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 6 GB (dedicated) | 8 GB+ |
| RAM       | 16 GB   | 32 GB |
| OS        | Windows 10/11 | Windows 11 |
| Disk      | 30 GB free for CARLA download | SSD |
| Python    | **3.11** (see note below) | 3.11 |

**Python version note:** CARLA 0.9.16 supports Python 3.7–3.12.
**You must use Python 3.11** if you have 3.13 or 3.14 installed, since the
`carla` pip package does not yet publish wheels for those versions.
Check with `py -0` on Windows.

---

## Setup

### Step 1 — Download and install CARLA

CARLA is a large standalone application, not a Python package.

1. Go to <https://carla.org/> and download the latest Windows precompiled
   release (currently 0.9.16).  The download is several GB.
2. Extract to a path **without spaces** — e.g. `C:\CARLA_0.9.16\`.
3. Launch CARLA by running `CarlaUE4.exe` (CARLA 0.9.x) from the extracted
   folder.  A window showing a driveable city should appear.
4. Leave that window running.  CARLA is now the **server**.

> **Windows quirk:** If you see a black screen or the window never renders,
> try launching from a terminal as Administrator, or add
> `-quality-level=Low` to the launch command for lower-end GPUs:
> `CarlaUE4.exe -quality-level=Low`

### Step 2 — Verify CARLA runs standalone

Before writing any code: confirm that `CarlaUE4.exe` opens a simulation window
and the city scene is visible.  If it crashes on startup, check your GPU
drivers and that you have ≥ 6 GB VRAM available (close other GPU-heavy apps).

### Step 3 — Create a Python 3.11 virtual environment

**Open a new terminal** (separate from the CARLA window — it must keep running).

```cmd
cd C:\Projects\AutoNav

:: Use Python 3.11 explicitly
py -3.11 -m venv venv

:: Activate
venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
```

The `carla==0.9.16` package must match your installed CARLA version exactly.
If you downloaded a different CARLA version, edit `requirements.txt`
accordingly before running `pip install`.

### Step 4 — Verify the connection (two-process model)

This is the most common point of confusion: **CARLA runs as a server, and
your Python script connects to it as a client.**  You must start them in order:

```
Terminal 1:   C:\CARLA_0.9.16\CarlaUE4.exe        ← start this first
Terminal 2:   py -3.11 src/verify_setup.py          ← then run this
```

Expected output in Terminal 2:
```
[verify] Connected to CARLA 0.9.16
[verify] Map: /Game/Carla/Maps/Town01
[verify] Vehicle spawned OK
[verify] Camera attached — waiting for first frame ...
[verify] Frame received: shape=(600, 800, 3)  dtype=uint8
[verify] ✓ Setup looks good! You can run main.py now.
```

If you see a connection error, CARLA is not running or is still loading.
Wait 10–20 seconds after `CarlaUE4.exe` opens, then retry.

---

## Running AutoNav

### Option A — Mock simulator (no CARLA needed, runs right now)

```cmd
cd C:\Projects\AutoNav
venv\Scripts\activate
py -3.11 src\main_sim.py --goal-x 100 --goal-y 100
```

Two windows open simultaneously:
- **AutoNav Dashboard** (OpenCV) — live camera feed, occupancy map with planned path, and the GT vs VO trajectory comparison
- **SimView** (pygame) — top-down bird's-eye view of the city grid, vehicle, buildings, and path

Press **Q** in either window (or **Ctrl-C** in the terminal) to stop.

| Flag | Default | Description |
|------|---------|-------------|
| `--goal-x` | `80.0` | Goal X in world metres |
| `--goal-y` | `120.0` | Goal Y in world metres |

### Option B — Real CARLA server

With `CarlaUE4.exe` already running in a separate terminal:

```cmd
venv\Scripts\activate
py -3.11 src\verify_setup.py          :: confirm connection first
py -3.11 src\main.py --map Town01 --goal-x 50 --goal-y 80
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | CARLA server address |
| `--port` | `2000` | CARLA server port |
| `--map` | `Town01` | CARLA map to load |
| `--goal-x` | `50.0` | Goal X in world metres |
| `--goal-y` | `80.0` | Goal Y in world metres |

Press **Q** in the dashboard window (or **Ctrl-C** in the terminal) to stop.

On clean shutdown, `results/trajectory_comparison.png` is saved automatically.

---

## Project Structure

```
autonav/
├── src/
│   ├── mock_carla/          Drop-in CARLA stub — runs without the real simulator
│   │   ├── __init__.py
│   │   ├── types.py         CARLA data types (Transform, VehicleControl, etc.)
│   │   └── world.py         Bicycle-model vehicle, vectorised camera + LiDAR
│   ├── sensors.py           Camera + LiDAR attachment and streaming
│   ├── visual_odometry.py   ORB feature tracking, Essential Matrix, pose integration
│   ├── mapping.py           Log-odds occupancy grid + Bresenham ray tracing
│   ├── path_planning.py     A* with obstacle inflation and path smoothing
│   ├── controller.py        Pure Pursuit + PID speed controller
│   ├── visualization.py     Live OpenCV dashboard (3 tiles)
│   ├── main_sim.py          Entry point — mock simulator (no CARLA needed)
│   ├── main.py              Entry point — real CARLA server
│   ├── verify_setup.py      One-shot CARLA connection + camera test
│   ├── smoke_test.py        Headless pipeline test (no display)
│   └── test_modules.py      Unit tests for all non-CARLA modules
├── results/
│   └── trajectory_comparison.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Overview

```
CARLA server
    │
    ├── RGB camera frames ──→ VisualOdometry (ORB + Essential Matrix)
    │                               │
    │                               ├──→ Estimated pose (x, y, heading)
    │                               │
    ├── LiDAR point cloud ─→ OccupancyGrid (log-odds, Bresenham)
    │                               │
    │                               └──→ 2-D obstacle map
    │                                         │
    │                                         └──→ A* Planner ──→ Waypoints
    │                                                                │
    ├── Ground-truth transform ──────────────────────────────→ PurePursuit
    │                                                                │
    └──────────────────────────────────────────────────────── VehicleControl
                                                            (steer, throttle, brake)
```

---

## Visual Odometry — Why It Drifts

Visual odometry estimates motion by tracking ORB keypoints across consecutive
camera frames and recovering rotation + translation from the Essential Matrix.

Every frame carries a small error (noisy feature localisation, RANSAC
outliers, near-degenerate scenes).  Because we *accumulate* these errors,
they compound over time — this is "drift", the fundamental limitation of
dead-reckoning systems.

You can observe this directly in the dashboard's trajectory tile: the green
(ground truth) and red (VO estimate) lines diverge as the drive progresses.
The terminal prints RMSE and final drift on exit.

**What full SLAM would add:** loop closure — recognising a previously-visited
place and using that constraint to retroactively correct the accumulated drift
across the entire map.  See "Documented Future Work" below.

---

## Documented Future Work

These are explicitly **not built** in this version.  They are honest future
directions, not bugs or omissions.

### Full SLAM with Loop Closure
Correct VO drift by recognising previously-visited locations (via a
visual place-recognition module such as DBoW2 or NetVLAD), building a
pose graph of constraints, and optimising it with g2o or GTSAM.  This is
research-grade complexity and a substantial separate project.

### Multi-Sensor Fusion (EKF)
Tightly fuse camera VO, LiDAR odometry, and CARLA's simulated IMU using an
Extended Kalman Filter.  This would dramatically reduce VO drift and is a
natural next step once single-sensor pipelines are solid.

### Dynamic Obstacle Avoidance
The current occupancy grid assumes static obstacles.  Supporting pedestrians
and other moving vehicles would require tracking detected objects over time
and reserving space in the map for their predicted future positions
(velocity obstacles or dynamic costmaps).

### Metric Scale Recovery
Monocular VO cannot recover metric scale without external cues.  This build
uses CARLA's ground-truth displacement to scale the VO translation each frame
(an honest research baseline).  Replacing this with stereo VO, or fusing with
a depth sensor, would make the pipeline fully self-contained.

---

## Known Windows Quirks

- CARLA must be on a path without spaces (e.g. `C:\CARLA_0.9.16\`).
- If CARLA hangs on startup, try disabling hardware ray tracing in the
  NVIDIA/AMD control panel, or launch with `-dx11` flag.
- Do not run CARLA inside a OneDrive-synced folder — file locking causes
  sporadic crashes.
- After closing CARLA, wait ~5 seconds before restarting it; the server port
  can remain in TIME_WAIT state briefly.
