"""
Particle filter on ScanNet: data access and the run entry point.

Usage:
    python -m sg2loc.scannet.particle_filter --config sg2loc/scannet/configs/val.yaml
"""

from __future__ import annotations

import os.path as osp

import numpy as np

from sg2loc.particle_filter.filter import ParticleFilter
from sg2loc.particle_filter.runner import run_main
from sg2loc.scannet import utils
from sg2loc.scannet.dataset import EVAL_SCANS_FILE, ScannetPatchObjDataset
from sg2loc.scannet.utils import ALIGNMENT_FILE, NATIVE_H, NATIVE_W


class ScannetParticleFilter(ParticleFilter):
    """Particle filter over the ScanNet layout: color/N.jpg, depth/N.png, pose/N.txt frames,
    precomputed _01 to _00 alignment, vertex-colored meshes (no texture)."""

    # landscape frames: gravity runs along the camera y-axis
    gravity_axis_col = 1

    def query_image_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(self.scans_scenes_dir, scan_id, "color", f"{frame_idx}.jpg")

    def load_query_poses(self, scan_id: str, frame_idxs) -> np.ndarray:
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

    def sensor_depth_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(self.scans_scenes_dir, scan_id, "depth", f"{frame_idx}.png")

    def floor_mask(self, vertices: np.ndarray, obj_ids: np.ndarray) -> np.ndarray:
        # gravity-aligned (z-up), so take the vertices just above the lowest height.
        floor_z = np.percentile(vertices[:, 2], 1)
        return vertices[:, 2] < floor_z + 0.15

    def map_texture_path(self, target_scan_id: str) -> None:
        return None  # vertex-colored mesh, rendered from per-triangle colors


def main() -> None:
    run_main(ScannetPatchObjDataset, ScannetParticleFilter, EVAL_SCANS_FILE)


if __name__ == "__main__":
    main()
