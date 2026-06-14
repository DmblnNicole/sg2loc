"""
Pose refinement on 3RScan: data access and the run entry point.

Usage:
    python -m sg2loc.scan3r.refinement --config sg2loc/scan3r/configs/val.yaml
        [--coarse-csv <sequence_poses_and_errors.csv>]
"""

from __future__ import annotations

import os.path as osp
from typing import Any

import numpy as np
from PIL import Image

from sg2loc.refinement.localizer import SequenceRefiner
from sg2loc.refinement.runner import run_main
from sg2loc.scan3r import utils
from sg2loc.scan3r.dataset import EVAL_SCANS_FILE, PatchObjectPairXTAESGIDataSet


class Scan3RSequenceRefiner(SequenceRefiner):
    """Refiner over the 3RScan layout: sequence/frame-XXXXXX.* frames, rescan alignment
    from 3RScan.json, views rendered by sampling the mesh UV texture."""

    def query_image_path(self, scan_id: str, frame_idx: str) -> str:
        return osp.join(
            self.scans_scenes_dir,
            scan_id,
            "sequence",
            f"frame-{str(frame_idx).zfill(6)}.color.jpg",
        )

    def load_query_poses(self, scan_id: str, frame_idxs) -> np.ndarray:
        return utils.load_all_poses(self.scans_scenes_dir, scan_id, frame_idxs)

    def map_to_query_transform(self, target_scan_id: str) -> np.ndarray:
        transforms = utils.load_rescan_transforms(
            osp.join(self.cfg.data.root_dir, "files/3RScan.json")
        )
        return transforms[target_scan_id]["transform_matrix"]

    def load_intrinsics(self, scan_id: str) -> np.ndarray:
        return utils.load_intrinsics(self.scans_scenes_dir, scan_id)["intrinsic_mat"]

    def hypothesis_axis(self) -> np.ndarray:
        # 3RScan frames are portrait: gravity runs along the camera x-axis
        return np.array([1.0, 0.0, 0.0])

    @staticmethod
    def load_appearance(cfg: Any, target_scan_id: str) -> np.ndarray:
        """The map mesh texture image as a (H, W, 3) uint8 array."""
        return np.asarray(
            Image.open(
                osp.join(cfg.particle_filter.scans_scenes_dir, target_scan_id, "mesh.refined_0.png")
            )
        )

    def render_view(self, uv_hits: np.ndarray, hit_ids: np.ndarray) -> np.ndarray:
        texture = self.appearance
        uv_hits[..., 1] = 1.0 - uv_hits[..., 1]  # flip v
        tex_h, tex_w, _ = texture.shape
        uv_pixels = (uv_hits * np.array([tex_w - 1, tex_h - 1])).astype(int)
        uv_pixels = np.clip(uv_pixels, 0, [tex_w - 1, tex_h - 1])
        color_img = texture[uv_pixels[..., 1], uv_pixels[..., 0]]
        return color_img.reshape(self.img_height, self.img_width, 3) / 255.0


def main() -> None:
    run_main(PatchObjectPairXTAESGIDataSet, Scan3RSequenceRefiner, EVAL_SCANS_FILE)


if __name__ == "__main__":
    main()
