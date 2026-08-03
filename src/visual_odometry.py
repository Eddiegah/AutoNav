"""
visual_odometry.py — Feature-based monocular visual odometry.

────────────────────────────────────────────────────────────────
WHAT THIS DOES (and why it drifts)
────────────────────────────────────────────────────────────────
Visual odometry (VO) estimates the vehicle's motion by tracking
distinctive image features across consecutive camera frames.

Pipeline per frame-pair:
  1.  Detect ORB keypoints in both frames.
  2.  Match descriptors with a brute-force Hamming matcher.
  3.  Filter matches with Lowe's ratio test (removes ambiguous matches).
  4.  Recover the Essential Matrix E from the filtered correspondences
      and the camera intrinsic matrix K.
  5.  Decompose E into rotation R and translation t.
      (Up to scale — monocular VO cannot recover metric scale without
       additional cues; we fix scale using CARLA's ground truth for the
       first frame, then let it free-run to keep the test honest.)
  6.  Accumulate R and t into a running pose estimate.

WHY DRIFT IS INEVITABLE (honest scope statement)
────────────────────────────────────────────────
Every frame-to-frame estimate carries a small error:
  • Noisy feature localisation (sub-pixel errors in matching).
  • Outliers that survive RANSAC inside findEssentialMat.
  • The Essential-Matrix decomposition is sensitive to planar or
    near-degenerate scenes.

Because we *accumulate* these errors (pose_k = pose_{k-1} * delta),
small per-frame errors compound over time — "drift" is simply
numerical integration error in disguise.

FULL SLAM would fix this via *loop closure*: recognising a
previously-visited place and using that constraint to correct the
accumulated drift across the entire trajectory.  Loop closure requires
a place-recognition module, a global pose graph, and a non-linear
optimiser (e.g. g2o, GTSAM).  That is genuinely research-grade
complexity and is explicitly out of scope for AutoNav — see README for
documented future work.

Drift is verified in realtime by comparing against CARLA's ground-truth
position; see evaluate_drift() below.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from sensors import build_camera_matrix


# ---------------------------------------------------------------------------
# VO configuration
# ---------------------------------------------------------------------------

# Number of ORB features to detect per frame.  More → slower but more
# robust in textureless scenes.
ORB_N_FEATURES = 2000

# Lowe's ratio-test threshold: discard a match if the best match distance
# is not sufficiently better than the second-best.  0.75 is the standard
# value from Lowe's SIFT paper.
RATIO_THRESH = 0.75

# Minimum inlier matches required to attempt pose recovery.
# Below this threshold the frame is skipped (e.g. motion blur, dark tunnel).
MIN_INLIERS = 30

# RANSAC reprojection threshold (pixels) inside findEssentialMat
RANSAC_THRESH = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Rigid body pose: 3×3 rotation and 3-vector translation (metres)."""
    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))

    def as_homogeneous(self) -> np.ndarray:
        """Return the 4×4 homogeneous transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3,  3] = self.t.ravel()
        return T

    @property
    def position_2d(self) -> tuple[float, float]:
        """
        (x, y) position on the ground plane in our world frame (metres).

        The VO translation vector t is expressed in the camera frame:
          t[0] = rightward (x in camera = right)
          t[1] = downward  (y in camera = down, ignored for 2-D ground plane)
          t[2] = forward   (z in camera = into scene = vehicle forward = our y)

        We map camera-frame (t[0], t[2]) → world-plane (x, y) to match
        the (x, y) convention used by OccupancyGrid and the CARLA/mock
        ground truth, which both report position as (location.x, location.y).
        """
        return float(self.t[0]), float(self.t[2])


# ---------------------------------------------------------------------------
# Visual odometry estimator
# ---------------------------------------------------------------------------

class VisualOdometry:
    """
    Incremental monocular visual odometry using ORB features.

    Usage:
        vo = VisualOdometry(K)
        for each new BGR frame:
            pose_estimate = vo.update(frame)
            x, z = pose_estimate.position_2d
    """

    def __init__(self, K: np.ndarray | None = None):
        # Camera intrinsic matrix (3×3)
        self.K: np.ndarray = K if K is not None else build_camera_matrix()

        # ORB detector — fast, binary descriptors, no patent issues
        self._orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)

        # Brute-force matcher using Hamming distance (correct for binary ORB)
        # crossCheck=False because we do Lowe's ratio test manually
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Previous frame state
        self._prev_frame: np.ndarray | None = None
        self._prev_kps:   list | None       = None
        self._prev_descs: np.ndarray | None = None

        # Accumulated pose (starts at origin)
        self._pose = Pose()

        # Scale from the previous frame (updated externally by ground truth
        # on the first real motion step)
        self._scale: float = 1.0

        # Statistics for drift evaluation
        self.frame_count: int = 0
        self.skipped_frames: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray,
               gt_scale: float | None = None) -> Pose:
        """
        Process one new BGR frame and return the updated pose estimate.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current camera frame (H × W × 3, BGR).
        gt_scale : float, optional
            If provided, override the translation scale for this step.
            Pass the Euclidean distance the vehicle moved according to
            CARLA's ground-truth transform; this removes the scale
            ambiguity inherent to monocular VO without making the
            drift comparison meaningless (R and t direction still come
            entirely from vision).

        Returns
        -------
        Pose
            Best current estimate of the vehicle's pose.
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)

        if self._prev_frame is None or descs is None or self._prev_descs is None:
            # Bootstrap: store first frame and return identity pose
            self._prev_frame = gray
            self._prev_kps   = kps
            self._prev_descs = descs
            return Pose(R=self._pose.R.copy(), t=self._pose.t.copy())

        # ── 1. Match descriptors ──────────────────────────────────────
        raw_matches = self._matcher.knnMatch(self._prev_descs, descs, k=2)

        # ── 2. Lowe's ratio test ─────────────────────────────────────
        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO_THRESH * n.distance:
                good.append(m)

        if len(good) < MIN_INLIERS:
            # Not enough reliable matches — skip this frame, keep old pose
            self.skipped_frames += 1
            self._prev_frame = gray
            self._prev_kps   = kps
            self._prev_descs = descs
            return Pose(R=self._pose.R.copy(), t=self._pose.t.copy())

        # ── 3. Build point correspondences ───────────────────────────
        pts_prev = np.float32(
            [self._prev_kps[m.queryIdx].pt for m in good])
        pts_curr = np.float32(
            [kps[m.trainIdx].pt for m in good])

        # ── 4. Essential Matrix with RANSAC ───────────────────────────
        # findEssentialMat internally enforces the epipolar constraint
        # and uses RANSAC to reject outlier matches.
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=RANSAC_THRESH,
        )

        if E is None or mask is None:
            self.skipped_frames += 1
            self._update_prev(gray, kps, descs)
            return Pose(R=self._pose.R.copy(), t=self._pose.t.copy())

        inlier_mask = mask.ravel().astype(bool)
        if inlier_mask.sum() < MIN_INLIERS:
            self.skipped_frames += 1
            self._update_prev(gray, kps, descs)
            return Pose(R=self._pose.R.copy(), t=self._pose.t.copy())

        # ── 5. Recover R, t from E ───────────────────────────────────
        # recoverPose returns R and t such that the translation vector
        # has unit norm (scale is lost in monocular VO).
        _, R_delta, t_delta, _ = cv2.recoverPose(
            E,
            pts_prev[inlier_mask],
            pts_curr[inlier_mask],
            cameraMatrix=self.K,
        )

        # ── 6. Apply scale ───────────────────────────────────────────
        # Use provided ground-truth scale if available; otherwise keep
        # the previous frame's scale (constant-velocity assumption).
        if gt_scale is not None and gt_scale > 0.01:
            self._scale = gt_scale
        t_scaled = t_delta * self._scale

        # ── 7. Integrate pose ────────────────────────────────────────
        # New global position = old_R * delta_t + old_t
        # New global rotation = old_R * delta_R
        self._pose.t = self._pose.R @ t_scaled + self._pose.t
        self._pose.R = self._pose.R @ R_delta

        self._update_prev(gray, kps, descs)
        return Pose(R=self._pose.R.copy(), t=self._pose.t.copy())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_prev(self, gray, kps, descs):
        self._prev_frame = gray
        self._prev_kps   = kps
        self._prev_descs = descs

    def reset(self):
        """Reset to identity pose (call when teleporting the vehicle)."""
        self._pose        = Pose()
        self._prev_frame  = None
        self._prev_kps    = None
        self._prev_descs  = None
        self._scale       = 1.0
        self.frame_count  = 0
        self.skipped_frames = 0

    @property
    def pose(self) -> Pose:
        return self._pose


# ---------------------------------------------------------------------------
# Drift evaluation helper
# ---------------------------------------------------------------------------

def evaluate_drift(estimated_positions: list[tuple[float, float]],
                   ground_truth_positions: list[tuple[float, float]]) -> dict:
    """
    Compare the estimated 2-D trajectory against CARLA's ground truth.

    Both lists contain (x, z) world-plane coordinates in metres,
    sampled at the same frame indices.

    Returns a dict with:
      'rmse'         — root-mean-square position error (metres)
      'max_error'    — worst single-frame error (metres)
      'final_drift'  — Euclidean error at the last frame (metres)
      'errors'       — per-frame error array
    """
    est = np.array(estimated_positions, dtype=np.float64)
    gt  = np.array(ground_truth_positions, dtype=np.float64)

    n = min(len(est), len(gt))
    if n == 0:
        return {"rmse": 0.0, "max_error": 0.0, "final_drift": 0.0, "errors": []}

    errors = np.linalg.norm(est[:n] - gt[:n], axis=1)
    return {
        "rmse":        float(np.sqrt(np.mean(errors ** 2))),
        "max_error":   float(np.max(errors)),
        "final_drift": float(errors[-1]),
        "errors":      errors.tolist(),
    }
