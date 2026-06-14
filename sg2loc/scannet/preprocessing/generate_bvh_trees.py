"""
Build the ScanNet BVH raycasting trees from mesh.

Usage:
    python -m sg2loc.scannet.preprocessing.generate_bvh_trees \
        --config sg2loc/scannet/configs/val.yaml --scans-dir /path/to/scannet/scans \
        --scene-list <txt> --output-dir /path/to/files/bvh_trees
"""

import argparse
import os
import os.path as osp

import numpy as np
import open3d as o3d

from sg2loc.configs import config, update_config
from sg2loc.preprocessing.bvh_build import (
    build_bvh,
    flatten_bvh,
    mean_edge_length,
    transfer_obj_ids,
)
from sg2loc.scannet.dataset import EVAL_SCANS_FILE


def build_scene(scan_dir: str, scan_id: str, output_dir: str, max_triangles_per_leaf: int) -> None:
    mesh = o3d.io.read_triangle_mesh(osp.join(scan_dir, f"{scan_id}_vh_clean_2.ply"))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.round(np.asarray(mesh.vertex_colors) * 255.0).astype(np.uint8)

    cloud = np.load(osp.join(scan_dir, "scene_graph_fusion", "data.npy"))
    cloud_points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(np.float64)
    cloud_ids = np.asarray(cloud["label"], dtype=np.int32)
    radius = 1.5 * mean_edge_length(vertices, triangles)
    obj_ids = transfer_obj_ids(vertices, cloud_points, None, cloud_ids, radius=radius)

    root = build_bvh(triangles, vertices, max_triangles_per_leaf)
    nodes: list = []
    leaf_triangles: list = []
    triangle_orig_indices: list = []
    flatten_bvh(root, nodes, leaf_triangles, triangle_orig_indices)

    os.makedirs(output_dir, exist_ok=True)
    np.save(osp.join(output_dir, "vertices.npy"), vertices.astype(np.float32))
    np.save(osp.join(output_dir, "bvh_nodes.npy"), nodes)
    np.save(osp.join(output_dir, "bvh_triangles.npy"), leaf_triangles)
    np.save(osp.join(output_dir, "obj_ids.npy"), obj_ids)
    np.save(osp.join(output_dir, "vertex_colors.npy"), vertex_colors)
    np.save(osp.join(output_dir, "triangle_orig_indices.npy"), triangle_orig_indices)
    # dummy texture arrays so the scene loader's interface stays uniform across datasets
    np.save(osp.join(output_dir, "uv_coords.npy"), np.zeros((len(vertices), 2), dtype=np.float32))
    np.save(osp.join(output_dir, "uv_indices.npy"), np.asarray(leaf_triangles, dtype=np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); the sg2loc.yaml overlay is merged automatically "
        "and provides max_triangles_per_leaf",
    )
    parser.add_argument(
        "--scans-dir", default="", help="ScanNet scans directory (default: from the config)"
    )
    parser.add_argument(
        "--scene-list", default="", help="txt with the query scans (default: the eval list)"
    )
    parser.add_argument(
        "--output-dir", default="", help="per-scan BVH root (default: from the config)"
    )
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    max_triangles_per_leaf = cfg.particle_filter.preprocess.max_triangles_per_leaf
    scans_dir = args.scans_dir or cfg.particle_filter.scans_scenes_dir
    output_root = args.output_dir or cfg.particle_filter.preprocess.output_dir
    scene_list = args.scene_list or str(EVAL_SCANS_FILE)

    scan_ids = [ln.strip() for ln in open(scene_list) if ln.strip()]
    # the list holds the _00 query scans, the BVH maps are their _01 twins
    scan_ids = [q[:-3] + "_01" for q in scan_ids]
    for i, scan_id in enumerate(scan_ids):
        out = osp.join(output_root, scan_id)
        if osp.isfile(osp.join(out, "vertex_colors.npy")):
            print(f"[bvh] skip (done) {scan_id}")
            continue
        build_scene(osp.join(scans_dir, scan_id), scan_id, out, max_triangles_per_leaf)
        print(f"[bvh] {i + 1}/{len(scan_ids)} {scan_id}")


if __name__ == "__main__":
    main()
