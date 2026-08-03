<div align="center">

# 🚗 AutoNav

### Perception-to-Path-Planning Autonomous Navigation Stack

*Visual Odometry · Occupancy Grid Mapping · A\* Path Planning · Pure Pursuit Control*

[![CI](https://github.com/Eddiegah/AutoNav/actions/workflows/ci.yml/badge.svg)](https://github.com/Eddiegah/AutoNav/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CARLA 0.9.16](https://img.shields.io/badge/CARLA-0.9.16-orange.svg)](https://carla.org/)

<br/>

> Built for **CARLA** — the open-source Unreal Engine autonomous driving simulator.  
> Runs **right now** with the built-in mock simulator. No CARLA download required to get started.

</div>

---

## What This Is

AutoNav is a complete robotics perception pipeline — the kind of stack that sits at the heart of every self-driving research prototype:

| Stage | What it does |
|---|---|
| **Sensors** | Streams RGB camera frames and LiDAR point clouds from a vehicle |
| **Visual Odometry** | Tracks ORB keypoints frame-to-frame, recovers rotation + translation via the Essential Matrix |
| **Occupancy Grid** | Builds a real-time 2-D obstacle map using log-odds updates and Bresenham ray tracing |
| **A\* Planner** | Finds the optimal path through the grid with costmap inflation for safety margins |
| **Pure Pursuit** | Geometric path-following controller with adaptive lookahead and PID speed control |
| **Dashboard** | Live OpenCV window — camera feed, occupancy map with path overlay, GT vs VO trajectory |

**Honest scope:** this is visual odometry + occupancy mapping + path planning. It is explicitly *not* full SLAM. There is no loop closure, no global bundle adjustment, no pose graph optimisation. Drift accumulates — that is real, expected, and honestly shown in the trajectory comparison panel. Full SLAM is documented as future work, not glossed over.

---

## Demo

<div align="center">

| Forward Camera | Occupancy Map + Path | GT vs VO Trajectory |
|:---:|:---:|:---:|
| Perspective raycaster with textured buildings, road markings, sky gradient | Local 120×120m view centred on vehicle, fills as LiDAR sweeps | Ground truth (green) vs VO estimate (red) drift comparison |

**Two windows run simultaneously:**
- `AutoNav Dashboard` (OpenCV) — the full sensor + planning pipeline view
- `AutoNav Top-Down View` (pygame) — bird's-eye city map with pulsing goal beacon, vehicle body, compass

</div>

---

## Quick Start

> **No CARLA download needed.** The built-in mock simulator runs everything locally.

```bash
# 1. Clone
git clone https://github.com/Eddiegah/AutoNav.git
cd AutoNav

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv          # Windows
# python3.11 -m venv venv      # Linux / macOS

# 3. Activate
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python src/main_sim.py --goal-x 100 --goal-y 100
```

Press **Q** in either window to stop. `results/trajectory_comparison.png` is saved on exit.

---

## Run Against Real CARLA

If you want the full photorealistic CARLA experience:

```bash
# Terminal 1 — start CARLA server first
C:\CARLA_0.9.16\CarlaUE4.exe

# Terminal 2 — verify connection, then run
python src/verify_setup.py
python src/main.py --map Town01 --goal-x 50 --goal-y 80
```

See [Setup → Real CARLA](#real-carla-setup) below for full instructions.

---

## Project Structure

```
AutoNav/
├── src/
│   ├── mock_carla/              Self-contained simulator (no CARLA needed)
│   │   ├── __init__.py
│   │   ├── types.py             CARLA API stubs (Transform, VehicleControl, …)
│   │   └── world.py             Bicycle-model vehicle · floor-cast camera · LiDAR
│   │
│   ├── sensors.py               Camera + LiDAR sensor attachment & streaming
│   ├── visual_odometry.py       ORB tracking → Essential Matrix → pose integration
│   ├── mapping.py               Log-odds occupancy grid + Bresenham ray tracing
│   ├── path_planning.py         A* with costmap inflation + path smoothing
│   ├── controller.py            Pure Pursuit steering + PID speed controller
│   ├── visualization.py         Dark-themed live OpenCV dashboard
│   │
│   ├── main_sim.py              Entry point — mock simulator  ← start here
│   ├── main.py                  Entry point — real CARLA server
│   ├── verify_setup.py          CARLA connection + camera frame test
│   ├── smoke_test.py            Headless end-to-end pipeline test
│   └── test_modules.py          Unit tests (no display, no server)
│
├── results/
│   └── trajectory_comparison.png
│
├── .github/
│   └── workflows/ci.yml         GitHub Actions CI (runs on every push)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How the Pipeline Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        CARLA / mock_carla                       │
│   RGB camera ──→ Visual Odometry (ORB + Essential Matrix)       │
│                        │                                        │
│                        ├──→  Estimated pose (x, y, heading)     │
│                        │                                        │
│   LiDAR sweep ──→ Occupancy Grid (log-odds + Bresenham)         │
│                        │                                        │
│                        └──→  A* Planner ──→ Waypoint list       │
│                                                │                │
│   Ground-truth transform ──→ Pure Pursuit ─────┘                │
│                                   │                             │
│                                   └──→ steer · throttle · brake │
└─────────────────────────────────────────────────────────────────┘
```

### Visual Odometry — and why it drifts

ORB features are detected in consecutive frames. Matched features feed into `findEssentialMat` (RANSAC inside) which recovers the rotation matrix **R** and unit translation **t**. These are accumulated:

```
pose_k = pose_{k-1} · [R | t·scale]
```

Every frame carries small errors — noisy feature localisation, RANSAC survivors, near-degenerate scenes. Because we *integrate* these errors, they compound. That is drift. The trajectory comparison panel in the dashboard shows it honestly in real time.

**What full SLAM would add:** loop closure — recognising a previously-visited location and using that constraint to retroactively correct the accumulated drift via a pose-graph optimiser (g2o / GTSAM). That is genuinely research-grade work and is listed as future work, not attempted here.

### A* Path Planning

```
f(n) = g(n) + h(n)
  g(n) = cost from start to n          (accumulated step cost)
  h(n) = Euclidean distance to goal    (admissible heuristic → optimal)
```

An inflation layer adds a cost penalty to cells within `INFLATION_RADIUS_CELLS` of any obstacle, so the planned path naturally stays away from walls. The resulting staircase path is smoothed with a rolling-average window before being handed to the controller.

### Pure Pursuit Controller

```
δ = atan( 2 · L · sin(α) / Ld )
  L   = wheelbase (2.9 m)
  α   = heading error to lookahead point
  Ld  = LOOKAHEAD_BASE + k · v   (adaptive — scales with speed)
```

A PID controller handles longitudinal speed. The derivative term damps oscillation on speed changes; the integral term corrects steady-state error on slopes.

---

## CLI Reference

### `main_sim.py` — mock simulator

```
python src/main_sim.py [--goal-x X] [--goal-y Y]

  --goal-x   Goal X coordinate in world metres  (default: 80.0)
  --goal-y   Goal Y coordinate in world metres  (default: 120.0)
```

### `main.py` — real CARLA

```
python src/main.py [--host H] [--port P] [--map M] [--goal-x X] [--goal-y Y]

  --host     CARLA server host   (default: 127.0.0.1)
  --port     CARLA server port   (default: 2000)
  --map      CARLA map name      (default: Town01)
  --goal-x   Goal X in metres    (default: 50.0)
  --goal-y   Goal Y in metres    (default: 80.0)
```

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| **Python** | 3.11 | 3.11 |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **RAM** | 8 GB | 16 GB |
| **GPU** *(mock sim)* | Any | Any |
| **GPU** *(real CARLA)* | 6 GB VRAM dedicated | 8 GB+ VRAM |
| **Disk** *(mock sim)* | ~500 MB | — |
| **Disk** *(real CARLA)* | 30 GB free | SSD |

---

## Real CARLA Setup

1. Download **CARLA 0.9.16** (Windows precompiled) from [carla.org](https://carla.org/)
2. Extract to `C:\CARLA_0.9.16\` — no spaces in path, not inside OneDrive
3. Launch `CarlaUE4.exe` and wait for the city window to appear
4. In a separate terminal with venv activated:

```bash
python src/verify_setup.py   # must print "Setup looks good!"
python src/main.py --map Town01 --goal-x 50 --goal-y 80
```

> **CARLA version note:** `carla==0.9.16` in `requirements.txt` must match your installed CARLA version exactly. The pip wheel only ships for Python 3.7–3.12 — Python 3.13+ is not supported.

> **Windows quirk:** if CARLA opens a black window, launch with `-quality-level=Low` or update GPU drivers.

---

## Documented Future Work

These are honest next steps, not hidden limitations:

- **Full SLAM with loop closure** — place recognition (DBoW2 / NetVLAD) + pose graph optimisation (g2o / GTSAM) to eliminate VO drift. Research-grade, substantial separate project.
- **Multi-sensor fusion (EKF)** — tightly fuse camera VO + LiDAR odometry + IMU for significantly more robust position estimates.
- **Dynamic obstacle avoidance** — the current occupancy grid treats all obstacles as static. Tracking moving objects (pedestrians, vehicles) with velocity obstacles or dynamic costmaps.
- **Metric scale recovery** — monocular VO has inherent scale ambiguity. This build uses CARLA's ground-truth displacement to set scale each frame. Stereo VO or depth-sensor fusion would make it fully self-contained.

---

## Running the Tests

```bash
# Unit tests — no display, no server
python src/test_modules.py

# End-to-end smoke test (50 sim ticks, headless)
python src/smoke_test.py
```

CI runs both automatically on every push via GitHub Actions.

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

Built with Python · OpenCV · NumPy · SciPy · pygame  
*"Don't describe this project as full SLAM — that would overclaim what's actually built."*

</div>
