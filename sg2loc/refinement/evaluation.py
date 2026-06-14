"""
Per-sequence pose error for the refinement stage, with a particle-filter fallback.

evaluate_sequence is the per-sequence core, shared with the inline path in localizer.py.
"""

from __future__ import annotations

import csv
import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


def calculate_pose_error(gt_pose: np.ndarray, localized_pose: np.ndarray) -> tuple:
    gt_rotation = gt_pose[:3, :3]
    localized_rotation = localized_pose[:3, :3]
    gt_translation = gt_pose[:3, 3]
    localized_translation = localized_pose[:3, 3]
    relative_rotation = localized_rotation @ gt_rotation.T
    # geodesic angle. Requires valid rotation matrices: a scaled or sheared
    # GT, e.g. from a non-rigid scan alignment. Otherwise error is wrong.
    rotation_error = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1.0, 1.0))
    positional_error = np.linalg.norm(gt_translation - localized_translation)
    return positional_error, np.degrees(rotation_error)


def parse_localized_pose_lines(lines) -> tuple:
    """Parse frame_poses.txt into (poses_dict, inliers_dict) keyed by scan_id_frame."""
    poses_dict = {}
    inliers_dict = {}
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        scan_id = parts[0]
        frame_name = f"{scan_id}_{parts[1]}"
        qx, qy, qz, qw = map(float, parts[2:6])
        tx, ty, tz = map(float, parts[6:9])
        inliers = int(parts[9])
        T = np.eye(4)
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        poses_dict[frame_name] = T
        inliers_dict[frame_name] = inliers
    return poses_dict, inliers_dict


def load_localized_poses(results_path: str) -> tuple:
    with open(results_path) as f:
        return parse_localized_pose_lines(f)


def backproject_to_last(
    localized_pose_raw: np.ndarray, max_inliers_gt_pose: np.ndarray, last_gt_pose: np.ndarray
) -> np.ndarray:
    """Propagate the localized pose of the max-inlier frame to the last query frame."""
    max_inliers_loc_pose = np.linalg.inv(localized_pose_raw)
    last_cam_T_max_cam = np.linalg.inv(max_inliers_gt_pose) @ last_gt_pose
    return max_inliers_loc_pose @ last_cam_T_max_cam


def choose_error(
    refined_pos: float,
    refined_rot: float,
    coarse_pos: float,
    coarse_rot: float,
    num_inliers: int,
    pnp_thresh: int,
) -> tuple:
    """Refined error when num_inliers > pnp_thresh, else coarse. Returns (pos, rot, used_refined)."""
    if num_inliers > pnp_thresh:
        return refined_pos, refined_rot, True
    return coarse_pos, coarse_rot, False


def build_coarse_lookup(coarse_df) -> dict:
    """Map (ScanID, FrameID) to its row of a particle-filter sequence_poses_and_errors.csv."""
    return {(str(r["ScanID"]), int(r["FrameID"])): r for _, r in coarse_df.iterrows()}


def evaluate_sequence(
    frame_idxs,
    gt_poses_rescan: dict,
    localized_raw: dict,
    inliers: dict,
    coarse_row,
    pnp_thresh: int,
) -> tuple | None:
    """Evaluate one sequence and return (pos, rot, used_refined), or None without a coarse row."""
    best = max(frame_idxs, key=lambda f: inliers[f])
    last = frame_idxs[-1]
    last_loc = backproject_to_last(
        localized_raw[best], gt_poses_rescan[best], gt_poses_rescan[last]
    )
    refined_pos, refined_rot = calculate_pose_error(gt_poses_rescan[last], last_loc)
    if coarse_row is None:
        return None
    return choose_error(
        refined_pos,
        refined_rot,
        coarse_row["PositionError"],
        coarse_row["RotationError"],
        inliers[best],
        pnp_thresh,
    )


def write_errors_csv(out_dir: str, errors: list) -> str:
    """Write the per-sequence [seq_idx, pos, rot] refined-pose errors. Returns the path."""
    path = os.path.join(out_dir, "sequence_errors.csv")
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "PositionalError", "RotationError (degrees)"])
        writer.writerows(errors)
    logger.info(f"Saved errors to {path}")
    return path
