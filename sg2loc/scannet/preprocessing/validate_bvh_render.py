"""
Validate the ScanNet BVH trees by rendering RGB through the CUDA raycaster.

Usage:
    python -m sg2loc.scannet.preprocessing.validate_bvh_render \
        --config sg2loc/scannet/configs/val.yaml --scans-dir <scans> \
        --bvh-dir <files/bvh_trees> --out-dir <dir> --scans scene0131_00 scene0653_01
"""

import argparse
import os
import os.path as osp

import numpy as np
from numba import cuda
from PIL import Image

from sg2loc.configs import config, update_config
from sg2loc.particle_filter.raycasting import ray_gen_and_traverse, ray_triangle_intersect_in_leaf
from sg2loc.particle_filter.scene import load_scene_geometry

RENDER_STRIDE = 2


def load_intrinsics(scan_dir: str) -> tuple:
    K = np.loadtxt(osp.join(scan_dir, "intrinsic", "intrinsic_color.txt"))[:3, :3]
    with Image.open(osp.join(scan_dir, "color", "0.jpg")) as im:
        width, height = im.size
    return K, width, height


def render(
    scene, K: np.ndarray, pose: np.ndarray, width: int, height: int, max_leaf_count: int
) -> np.ndarray:
    w, h = width // RENDER_STRIDE, height // RENDER_STRIDE
    num_rays = w * h
    num_columns = 1 + max_leaf_count + 7
    ray_leaf_info = cuda.to_device(np.zeros((num_rays, num_columns), dtype=np.float32))
    uv_hits = cuda.to_device(np.zeros((num_rays, 2), dtype=np.float32))
    hit_record = cuda.to_device(np.full(num_rays, -1, dtype=np.int32))
    depth_map = cuda.to_device(np.zeros(num_rays, dtype=np.float32))
    threads = 256
    blocks = (num_rays + threads - 1) // threads
    K_gpu = cuda.to_device(np.ascontiguousarray(K.astype(np.float64)))
    Rs = cuda.to_device(pose[:3, :3][None].astype(np.float32))
    ts = cuda.to_device(pose[:3, 3][None].astype(np.float32))
    ray_gen_and_traverse[blocks, threads](
        K_gpu, Rs, ts, scene.bvh_nodes_gpu, ray_leaf_info, width, height, RENDER_STRIDE, 1
    )
    ray_triangle_intersect_in_leaf[blocks, threads](
        ray_leaf_info,
        scene.bvh_nodes_gpu,
        scene.bvh_triangles_gpu,
        scene.vertices_gpu,
        scene.uv_coords_gpu,
        scene.uv_indices_gpu,
        scene.triangle_orig_indices_gpu,
        uv_hits,
        hit_record,
        depth_map,
        num_rays,
    )
    hits = hit_record.copy_to_host()
    tri_colors = scene.vertex_colors[scene.bvh_triangles].mean(axis=1).astype(np.uint8)
    img = np.full((num_rays, 3), 40, dtype=np.uint8)
    valid = hits >= 0
    img[valid] = tri_colors[hits[valid]]
    return img.reshape(h, w, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); provides observation_model.max_leaf_count",
    )
    parser.add_argument("--scans-dir", required=True)
    parser.add_argument("--bvh-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scans", nargs="+", required=True)
    parser.add_argument("--frame", type=int, default=0, help="frame index to render")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    max_leaf_count = cfg.particle_filter.observation_model.max_leaf_count

    os.makedirs(args.out_dir, exist_ok=True)
    for scan_id in args.scans:
        scan_dir = osp.join(args.scans_dir, scan_id)
        scene = load_scene_geometry(osp.join(args.bvh_dir, scan_id))
        scene.vertex_colors = np.load(osp.join(args.bvh_dir, scan_id, "vertex_colors.npy"))
        scene.bvh_triangles = np.load(osp.join(args.bvh_dir, scan_id, "bvh_triangles.npy"))
        K, width, height = load_intrinsics(scan_dir)
        pose = np.loadtxt(osp.join(scan_dir, "pose", f"{args.frame}.txt"))
        rendered = render(scene, K, pose, width, height, max_leaf_count)
        query = np.asarray(
            Image.open(osp.join(scan_dir, "color", f"{args.frame}.jpg")).resize(
                (rendered.shape[1], rendered.shape[0])
            )
        )
        pair = np.concatenate([query, rendered], axis=1)
        out = osp.join(args.out_dir, f"{scan_id}_{args.frame}.png")
        Image.fromarray(pair).save(out)
        print(f"{scan_id}: hit {(rendered != 40).any(axis=-1).mean() * 100:.0f}% -> {out}")


if __name__ == "__main__":
    main()
