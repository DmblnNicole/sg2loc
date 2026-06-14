"""
Cache each scans aligned annotated mesh (labels.instances.align.annotated.v2.ply) as
scenes/<scan_id>/data.npy and its point-cloud center as pcl_center.npy.

Usage:
    python -m sg2loc.scan3r.preprocessing.annotated_ply_to_npy \
        --config sg2loc/scan3r/configs/val.yaml
"""

import argparse
import logging
import os
import os.path as osp

import numpy as np
from plyfile import PlyData

from sg2loc.configs import config, update_config
from sg2loc.utils import point_cloud

logger = logging.getLogger(__name__)

ANNOTATED_MESH = "labels.instances.align.annotated.v2.ply"
CACHE_NAME = "data.npy"
CENTER_NAME = "pcl_center.npy"


def convert_scan(scene_dir: str, force: bool) -> str:
    cache_path = osp.join(scene_dir, CACHE_NAME)
    center_path = osp.join(scene_dir, CENTER_NAME)
    result = "kept"
    if not osp.exists(cache_path) or force:
        ply_path = osp.join(scene_dir, ANNOTATED_MESH)
        if not osp.exists(ply_path):
            return "no annotated mesh"
        vertices = np.asarray(PlyData.read(ply_path)["vertex"].data)
        # the established cache stores the ids signed (the ply declares them unsigned)
        target = np.dtype(
            [
                (name, "<i2" if name in ("objectId", "globalId") else vertices.dtype[name])
                for name in vertices.dtype.names
            ]
        )
        assert vertices["objectId"].max() < 2**15 and vertices["globalId"].max() < 2**15
        np.save(cache_path, vertices.astype(target))
        result = "written"
    if not osp.exists(center_path) or force:
        points = point_cloud.load_plydata_npy(cache_path)
        np.save(center_path, np.mean(points, axis=0))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str, help="configuration file name")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing caches")
    args = parser.parse_args()
    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)

    scenes_dir = cfg.particle_filter.scans_scenes_dir
    counts: dict = {}
    for scan_id in sorted(os.listdir(scenes_dir)):
        scene_dir = osp.join(scenes_dir, scan_id)
        if not osp.isdir(scene_dir):
            continue
        result = convert_scan(scene_dir, args.force)
        counts[result] = counts.get(result, 0) + 1
    logger.info(", ".join(f"{n} {result}" for result, n in sorted(counts.items())))


if __name__ == "__main__":
    main()
