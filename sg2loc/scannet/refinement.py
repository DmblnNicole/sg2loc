"""
Pose refinement on ScanNet: data access and the run entry point.

Usage:
    python -m sg2loc.scannet.refinement --config sg2loc/scannet/configs/val.yaml
        [--coarse-csv <sequence_poses_and_errors.csv>]
"""

from __future__ import annotations

import os.path as osp
from typing import Any

import numpy as np

from sg2loc.refinement.localizer import SequenceRefiner
from sg2loc.refinement.runner import run_main
from sg2loc.scannet import utils
from sg2loc.scannet.dataset import EVAL_SCANS_FILE, ScannetPatchObjDataset
from sg2loc.scannet.utils import ALIGNMENT_FILE, NATIVE_H, NATIVE_W


class ScannetSequenceRefiner(SequenceRefiner):
    """Refiner over the ScanNet layout: color/N.jpg, pose/N.txt frames, precomputed
    _01 to _00 alignment, views rendered from vertex colors."""

    def query_image_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(self.scans_scenes_dir, scan_id, "color", f"{frame_idx}.jpg")

    def load_query_poses(self, scan_id: str, frame_idxs) -> np.ndarray:
        # load only the requested frames
        pose_dir = osp.join(self.scans_scenes_dir, scan_id, "pose")
        return np.array(
            [np.loadtxt(osp.join(pose_dir, f"{frame_idx}.txt")) for frame_idx in frame_idxs]
        )

    def map_to_query_transform(self, target_scan_id: str) -> np.ndarray:
        return utils.load_db_to_query_transforms(ALIGNMENT_FILE)[target_scan_id]

    def load_intrinsics(self, scan_id: str) -> np.ndarray:
        # scale the native-resolution intrinsics to the configured render resolution
        K = utils.load_frame_intrinsics(self.scans_scenes_dir, scan_id).copy()
        K[0] *= self.cfg.data.img.w / NATIVE_W
        K[1] *= self.cfg.data.img.h / NATIVE_H
        return K

    def hypothesis_axis(self) -> np.ndarray:
        # ScanNet frames are landscape: gravity runs along the camera y-axis
        return np.array([0.0, 1.0, 0.0])

    @staticmethod
    def load_appearance(cfg: Any, target_scan_id: str) -> tuple:
        """Per-vertex colors and triangle indices for barycentric vertex-color rendering."""
        bvh_dir = osp.join(cfg.particle_filter.preprocess.output_dir, target_scan_id)
        vertex_colors = np.load(osp.join(bvh_dir, "vertex_colors.npy")).astype(np.float32)
        bvh_triangles = np.load(osp.join(bvh_dir, "bvh_triangles.npy"))
        return vertex_colors, bvh_triangles

    def render_view(self, uv_hits: np.ndarray, hit_ids: np.ndarray) -> np.ndarray:
        # barycentric vertex-color interpolation, uv_hits carries the hit (u, v) barycentrics
        vertex_colors, bvh_triangles = self.appearance
        color_img = np.zeros((hit_ids.shape[0], 3), dtype=np.float32)  # misses render black
        valid = hit_ids >= 0
        tris = bvh_triangles[hit_ids[valid]]  # (N, 3) vertex indices
        uv = uv_hits.reshape(-1, 2)[valid]
        w = 1.0 - uv[:, 0] - uv[:, 1]
        color_img[valid] = (
            w[:, None] * vertex_colors[tris[:, 0]]
            + uv[:, 0:1] * vertex_colors[tris[:, 1]]
            + uv[:, 1:2] * vertex_colors[tris[:, 2]]
        )
        return color_img.reshape(self.img_height, self.img_width, 3) / 255.0


def main() -> None:
    run_main(ScannetPatchObjDataset, ScannetSequenceRefiner, EVAL_SCANS_FILE)


if __name__ == "__main__":
    main()
