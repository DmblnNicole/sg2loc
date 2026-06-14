"""
Precompute the per-scene BVH trees for the 3RScan particle filter (evaluation maps).

Builds a BVH over Blender-decimated mesh (low.obj) for each scene and writes seven .npy files to
<preprocess.output_dir>/<scan_id>/ that the CUDA raycaster consumes:

    vertices.npy              (V, 3) float64   low.obj vertices
    bvh_nodes.npy             (N, 10) float64  flattened BVH nodes (see bvh_build.NODE_NUM_COLUMNS)
    bvh_triangles.npy         (T, 3) int32     triangles in leaf-traversal order
    obj_ids.npy               (V,)   int32     per-vertex object id (3-NN vote vs annotated mesh)
    uv_coords.npy             (3T, 2) float64  low.obj texture coords, 3 per triangle
    uv_indices.npy            (T, 3) int64     index into uv_coords per triangle
    triangle_orig_indices.npy (T,)   int64     leaf-order -> original-order permutation

Geometry and UVs come from low.obj. Object ids are transferred from the annotated mesh
(labels.instances.annotated.v2.ply plus data.npy) by a radius-limited 3-NN vote.

Usage:
    python -m sg2loc.scan3r.preprocessing.generate_bvh_trees \
        --config sg2loc/scan3r/configs/val.yaml \
        --scan-list sg2loc/scan3r/preprocessing/scene_lists/scan3r_eval_rescans.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp

import numpy as np
import open3d as o3d
from tqdm import tqdm

from sg2loc.configs import config, update_config
from sg2loc.preprocessing.bvh_build import (
    build_bvh,
    flatten_bvh,
    transfer_obj_ids,
)

logger = logging.getLogger(__name__)

DECIMATED_MESH = "low.obj"
ANNOTATED_MESH = "labels.instances.annotated.v2.ply"
OBJ_ID_FILE = "data.npy"


def build_scene_bvh(scene_dir: str, output_dir: str, max_triangles_per_leaf: int) -> None:
    mesh = o3d.io.read_triangle_mesh(
        osp.join(scene_dir, DECIMATED_MESH), enable_post_processing=True
    )
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    uv_coords = np.asarray(mesh.triangle_uvs)  # (3T, 2), 3 per triangle
    uv_indices = np.arange(len(uv_coords)).reshape(-1, 3)  # (T, 3)

    annotated = o3d.io.read_triangle_mesh(osp.join(scene_dir, ANNOTATED_MESH))
    annotated_obj_ids = np.asarray(
        np.load(osp.join(scene_dir, OBJ_ID_FILE))["objectId"], dtype=np.int32
    )
    obj_ids = transfer_obj_ids(
        vertices,
        np.asarray(annotated.vertices),
        np.asarray(annotated.triangles),
        annotated_obj_ids,
    )

    root = build_bvh(triangles, vertices, max_triangles_per_leaf)
    nodes: list = []
    leaf_triangles: list = []
    triangle_orig_indices: list = []
    flatten_bvh(root, nodes, leaf_triangles, triangle_orig_indices)

    os.makedirs(output_dir, exist_ok=True)
    np.save(osp.join(output_dir, "vertices.npy"), vertices)
    np.save(osp.join(output_dir, "bvh_nodes.npy"), nodes)
    np.save(osp.join(output_dir, "bvh_triangles.npy"), leaf_triangles)
    np.save(osp.join(output_dir, "obj_ids.npy"), obj_ids)
    np.save(osp.join(output_dir, "uv_coords.npy"), uv_coords)
    np.save(osp.join(output_dir, "uv_indices.npy"), uv_indices)
    np.save(osp.join(output_dir, "triangle_orig_indices.npy"), triangle_orig_indices)


def _read_scan_list(path: str | None) -> set | None:
    if path is None:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)  # silence CUDA dealloc chatter
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); the sg2loc.yaml overlay is merged "
        "automatically. Provides scenes dir, output dir and "
        "max_triangles_per_leaf.",
    )
    parser.add_argument(
        "--scan-list",
        default=None,
        help="File with one scan_id per line; default: all scenes that "
        "contain a low.obj (the evaluation rescans).",
    )
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    scenes_dir = cfg.particle_filter.scans_scenes_dir
    output_root = cfg.particle_filter.preprocess.output_dir
    max_triangles_per_leaf = cfg.particle_filter.preprocess.max_triangles_per_leaf

    wanted = _read_scan_list(args.scan_list)
    scan_ids = sorted(
        name
        for name in os.listdir(scenes_dir)
        if osp.isdir(osp.join(scenes_dir, name)) and (wanted is None or name in wanted)
    )

    built = []
    for scan_id in tqdm(scan_ids, desc="BVH"):
        scene_dir = osp.join(scenes_dir, scan_id)
        if not osp.exists(osp.join(scene_dir, DECIMATED_MESH)):
            continue
        build_scene_bvh(scene_dir, osp.join(output_root, scan_id), max_triangles_per_leaf)
        built.append(scan_id)

    logger.info(f"built {len(built)} scene(s)")
    if wanted is not None:
        missing = sorted(wanted - set(built))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} listed scan(s) could not be built (missing scene dir or "
                f"low.obj): {missing}. The evaluation map set would be incomplete."
            )


if __name__ == "__main__":
    main()
