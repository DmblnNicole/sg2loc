"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG

Usage:
    python -m sg2loc.scan3r.preprocessing.align_annotated_ply \
        --config sg2loc/scan3r/configs/val.yaml
"""

import argparse
import json
import logging
import os
import os.path as osp
from shutil import copyfile

import numpy as np
from plyfile import PlyData

from sg2loc.configs import config, update_config

logger = logging.getLogger(__name__)

RAW_MESH = "labels.instances.annotated.v2.ply"
ALIGNED_MESH = "labels.instances.align.annotated.v2.ply"


def read_rescan_transforms(scan_config_file: str) -> dict:
    # in 3RScan.json the reference field of a rescan entry holds the rescan id
    rescan2ref = {}
    with open(scan_config_file) as f:
        for scene in json.load(f):
            for scan in scene["scans"]:
                if "transform" in scan:
                    rescan2ref[scan["reference"]] = np.matrix(scan["transform"]).reshape(4, 4)
    return rescan2ref


def resave_ply(filename_in: str, filename_out: str, matrix: np.matrix) -> None:
    plydata = PlyData.read(open(filename_in, "rb"))
    points = np.stack(
        (plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"])
    ).transpose()
    points4f = np.insert(points, 3, values=1, axis=1)
    points = points4f * matrix
    plydata["vertex"]["x"] = np.asarray(points[:, 0]).flatten()
    plydata["vertex"]["y"] = np.asarray(points[:, 1]).flatten()
    plydata["vertex"]["z"] = np.asarray(points[:, 2]).flatten()
    plydata.write(filename_out)


def align_scan(scene_dir: str, rescan2ref: dict, force: bool) -> str:
    scan_id = osp.basename(scene_dir)
    file_in = osp.join(scene_dir, RAW_MESH)
    file_out = osp.join(scene_dir, ALIGNED_MESH)
    if osp.isfile(file_out) and not force:
        return "kept"
    if not osp.isfile(file_in):
        return "no annotated mesh"
    if scan_id in rescan2ref:
        resave_ply(file_in, file_out, rescan2ref[scan_id])
        return "aligned"
    copyfile(file_in, file_out)
    return "copied"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str, help="configuration file name")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing meshes")
    args = parser.parse_args()
    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)

    rescan2ref = read_rescan_transforms(osp.join(cfg.data.root_dir, "files/3RScan.json"))
    scenes_dir = cfg.particle_filter.scans_scenes_dir
    counts: dict = {}
    for scan_id in sorted(os.listdir(scenes_dir)):
        scene_dir = osp.join(scenes_dir, scan_id)
        if not osp.isdir(scene_dir):
            continue
        result = align_scan(scene_dir, rescan2ref, args.force)
        counts[result] = counts.get(result, 0) + 1
    logger.info(", ".join(f"{n} {result}" for result, n in sorted(counts.items())))


if __name__ == "__main__":
    main()
