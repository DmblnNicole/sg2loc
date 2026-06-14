"""
Particle filter on 3RScan: data access and the run entry point.

Usage:
    python -m sg2loc.scan3r.particle_filter --config sg2loc/scan3r/configs/val.yaml
"""

from __future__ import annotations

import os.path as osp

import numpy as np

from sg2loc.particle_filter.filter import ParticleFilter
from sg2loc.particle_filter.runner import run_main
from sg2loc.scan3r import utils
from sg2loc.scan3r.dataset import EVAL_SCANS_FILE, PatchObjectPairXTAESGIDataSet
from sg2loc.scan3r.visualize import render_sequence_gif


class Scan3RParticleFilter(ParticleFilter):
    """Particle filter over the 3RScan layout: sequence/frame-XXXXXX.* frames, rescan
    alignment from 3RScan.json, UV-textured meshes."""

    # gravity runs along the camera x-axis
    gravity_axis_col = 0

    def query_image_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(self.scans_scenes_dir, scan_id, "sequence", f"frame-{frame_idx}.color.jpg")

    def load_query_poses(self, scan_id: str, frame_idxs) -> np.ndarray:
        return utils.load_all_poses(self.scans_scenes_dir, scan_id, frame_idxs)

    def map_to_query_transform(self, target_scan_id: str) -> np.ndarray:
        transforms = utils.load_rescan_transforms(
            osp.join(self.cfg.data.root_dir, "files/3RScan.json")
        )
        return transforms[target_scan_id]["transform_matrix"]

    def load_intrinsics(self, scan_id: str) -> np.ndarray:
        return utils.load_intrinsics(self.scans_scenes_dir, scan_id)["intrinsic_mat"]

    def sensor_depth_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(self.scans_scenes_dir, scan_id, "sequence", f"frame-{frame_idx}.depth.pgm")

    def floor_mask(self, vertices: np.ndarray, obj_ids: np.ndarray) -> np.ndarray:
        # the 3RScan GT annotation labels the floor as object id 1
        return obj_ids == 1

    def map_texture_path(self, target_scan_id: str) -> str:
        return osp.join(self.scans_scenes_dir, target_scan_id, "mesh.refined_0.png")


def main() -> None:
    run_main(
        PatchObjectPairXTAESGIDataSet,
        Scan3RParticleFilter,
        EVAL_SCANS_FILE,
        visualize_fn=render_sequence_gif,
    )


if __name__ == "__main__":
    main()
