"""
path_planning.py — A* path planning on the occupancy grid.

────────────────────────────────────────────────────────────────
WHY A* (and how it works)
────────────────────────────────────────────────────────────────
A* is a best-first graph search algorithm that finds the shortest
path between two nodes in a weighted graph.  It is the right choice
here because:

  • It is guaranteed to find the optimal (shortest) path if the
    heuristic is *admissible* — i.e. it never *overestimates* the
    true cost.  We use the Euclidean distance to the goal, which
    is always ≤ the true grid path length.
  • It is significantly faster than Dijkstra's algorithm (which
    explores in all directions equally) because the heuristic steers
    the search toward the goal.
  • It is well-understood, easy to verify, and maps naturally onto
    a 2-D grid.

Core concepts
─────────────
  f(n) = g(n) + h(n)

  g(n)  cost from the start cell to cell n (accumulated step cost)
  h(n)  heuristic estimate of cost from n to the goal (Euclidean dist)
  f(n)  total estimated cost of the cheapest path through n

  The open set (priority queue, min-heap on f) always pops the
  cell with the lowest f — i.e. the most promising next candidate.
  The closed set tracks cells already settled (optimal cost known).

Step costs:
  • Moving to an axis-aligned neighbour costs 1 cell.
  • Moving diagonally costs √2 ≈ 1.414 cells.
  • Moving into a cell near an obstacle incurs an extra *inflation*
    penalty — this creates a safety margin so the planned path stays
    away from walls (analogous to the costmap inflation layer in ROS).

Re-planning:
  The planner is called every N frames from the main loop.  When
  new obstacles appear in the occupancy grid the old path may cross
  occupied cells; re-running A* from the current cell gives a fresh
  path that avoids them.

────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import heapq
import numpy as np
from mapping import OccupancyGrid, GRID_CELLS, GRID_RESOLUTION


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Extra cost multiplier applied to cells adjacent to occupied cells.
# Higher values keep the path further from obstacles.
INFLATION_COST = 5.0

# Number of grid cells to inflate around each obstacle
INFLATION_RADIUS_CELLS = 3

# Diagonal movement allowed?
ALLOW_DIAGONAL = True


# ---------------------------------------------------------------------------
# Pre-compute neighbour offsets
# ---------------------------------------------------------------------------

if ALLOW_DIAGONAL:
    _NEIGHBORS = [
        (-1,  0, 1.0),    # up
        ( 1,  0, 1.0),    # down
        ( 0, -1, 1.0),    # left
        ( 0,  1, 1.0),    # right
        (-1, -1, 1.414),  # diagonals
        (-1,  1, 1.414),
        ( 1, -1, 1.414),
        ( 1,  1, 1.414),
    ]
else:
    _NEIGHBORS = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
    ]


# ---------------------------------------------------------------------------
# Costmap builder
# ---------------------------------------------------------------------------

def _build_costmap(grid: OccupancyGrid) -> np.ndarray:
    """
    Build a float32 costmap from the binary occupancy grid.

    Obstacle cells → np.inf (impassable)
    Cells within INFLATION_RADIUS_CELLS of an obstacle → INFLATION_COST
    All other cells → 1.0 (unit cost)

    Using a costmap (rather than querying the grid per-step) speeds up
    A* significantly — we pay for the inflation pass once per plan call.
    """
    binary = grid.get_binary().astype(np.float32) / 255.0  # 0 or 1

    # Mark obstacles as impassable
    costmap = np.ones((GRID_CELLS, GRID_CELLS), dtype=np.float32)
    costmap[binary > 0.5] = np.inf

    # Simple inflation: dilate obstacle mask and add penalty
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * INFLATION_RADIUS_CELLS + 1,
                      2 * INFLATION_RADIUS_CELLS + 1), dtype=bool)
    inflated = binary_dilation(binary > 0.5, structure=struct)
    near_obstacle = inflated & (binary <= 0.5)   # inflated but not occupied
    costmap[near_obstacle] += INFLATION_COST

    return costmap


# ---------------------------------------------------------------------------
# A* implementation
# ---------------------------------------------------------------------------

def _heuristic(r: int, c: int, gr: int, gc: int) -> float:
    """
    Euclidean distance heuristic — admissible (never overestimates)
    because the straight-line distance is always ≤ the actual grid path.
    """
    return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5


def astar(grid: OccupancyGrid,
          start_world: tuple[float, float],
          goal_world:  tuple[float, float]
          ) -> list[tuple[float, float]] | None:
    """
    Plan a path from *start_world* to *goal_world* (both in world
    coordinates, metres) using A* on *grid*.

    Returns
    -------
    list of (x, y) world-coordinate waypoints if a path was found,
    or None if the goal is unreachable (surrounded by obstacles, or
    outside the grid).

    The returned path is smoothed with _smooth_path() so the vehicle
    does not make unnecessarily sharp turns at every grid cell.
    """
    # Convert world positions to grid cells
    sr, sc = grid.world_to_cell(*start_world)
    gr, gc = grid.world_to_cell(*goal_world)

    # Bounds / sanity checks
    if not (_in_bounds(sr, sc) and _in_bounds(gr, gc)):
        print("[PathPlanner] Start or goal is outside the grid bounds.")
        return None
    if grid.is_occupied(*goal_world):
        print("[PathPlanner] Goal cell is marked occupied — cannot plan.")
        return None

    costmap = _build_costmap(grid)

    if costmap[sr, sc] == np.inf:
        print("[PathPlanner] Start cell is occupied — shifting to nearest free.")
        sr, sc = _find_nearest_free(costmap, sr, sc)
        if sr is None:
            return None

    # ── A* ──────────────────────────────────────────────────────────
    # open_heap entries: (f_score, g_score, row, col)
    open_heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (0.0 + _heuristic(sr, sc, gr, gc), 0.0, sr, sc))

    g_score: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        # ── Goal reached ─────────────────────────────────────────
        if r == gr and c == gc:
            return _reconstruct_and_smooth(came_from, grid, gr, gc)

        # ── Expand neighbours ─────────────────────────────────────
        for dr, dc, step_cost in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc):
                continue
            if (nr, nc) in closed:
                continue
            cell_cost = costmap[nr, nc]
            if cell_cost == np.inf:
                continue   # impassable obstacle

            new_g = g + step_cost * cell_cost
            if new_g < g_score.get((nr, nc), np.inf):
                g_score[(nr, nc)]   = new_g
                came_from[(nr, nc)] = (r, c)
                f_new = new_g + _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_heap, (f_new, new_g, nr, nc))

    print("[PathPlanner] A* exhausted open set — no path found.")
    return None


# ---------------------------------------------------------------------------
# Path reconstruction and smoothing
# ---------------------------------------------------------------------------

def _reconstruct_and_smooth(came_from, grid, gr, gc
                             ) -> list[tuple[float, float]]:
    """Walk the came_from dict to build the raw path, then smooth it."""
    path_cells: list[tuple[int, int]] = []
    node: tuple[int, int] | None = (gr, gc)
    while node is not None:
        path_cells.append(node)
        node = came_from[node]
    path_cells.reverse()

    # Convert cells → world coordinates (centre of each cell)
    waypoints_world = [grid.cell_to_world(r, c) for r, c in path_cells]

    return _smooth_path(waypoints_world)


def _smooth_path(waypoints: list[tuple[float, float]],
                 window: int = 5) -> list[tuple[float, float]]:
    """
    Simple moving-average path smoothing.

    A* paths on grids are "staircase" shaped; averaging over a
    rolling window rounds the corners and reduces unnecessary steering
    commands.  The start and goal are preserved exactly.
    """
    if len(waypoints) <= 2 * window:
        return waypoints

    xs = np.array([p[0] for p in waypoints])
    ys = np.array([p[1] for p in waypoints])
    kernel = np.ones(window) / window
    xs_s = np.convolve(xs, kernel, mode="same")
    ys_s = np.convolve(ys, kernel, mode="same")

    # Restore exact start and end
    xs_s[0],  ys_s[0]  = xs[0],  ys[0]
    xs_s[-1], ys_s[-1] = xs[-1], ys[-1]

    return list(zip(xs_s.tolist(), ys_s.tolist()))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GRID_CELLS and 0 <= c < GRID_CELLS


def _find_nearest_free(costmap: np.ndarray,
                       sr: int, sc: int,
                       search_radius: int = 10
                       ) -> tuple[int | None, int | None]:
    """BFS outward from (sr, sc) to find the nearest free cell."""
    from collections import deque
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    visited = {(sr, sc)}
    while q:
        r, c = q.popleft()
        if costmap[r, c] < np.inf:
            return r, c
        for dr, dc, _ in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc) and (nr, nc) not in visited:
                if abs(nr - sr) <= search_radius and abs(nc - sc) <= search_radius:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return None, None
