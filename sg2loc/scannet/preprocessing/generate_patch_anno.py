"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.

Usage:
    python -m sg2loc.scannet.preprocessing.generate_patch_anno \
        --config sg2loc/scannet/configs/val.yaml --scans-dir /path/to/scannet/scans \
        --files-dir /path/to/scannet/files --scene-list <query scans txt>
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
from tqdm import tqdm

from sg2loc.configs import config, update_config
from sg2loc.scannet.utils import NATIVE_H, NATIVE_W, load_frame_idxs, load_frame_intrinsics

ANNO_RENDER_W = 448  # annotation raycast resolution
ANNO_RENDER_H = 336
LABEL_TRANSFER_DIST_TH = 0.05  # m, max cloud-to-mesh distance for the label transfer
PATCH_VOTE_TH = 0.2  # min fraction of patch pixels agreeing on the winning object id
MESH_NAME = "_vh_clean_2.labels.ply"
SGFUSION_PLY = "scene_graph_fusion/inseg_filtered.ply"


def raycast_frame(scene, mesh_triangles, intrinsic, pose_C_W, width, height):
    """Hit mask and per-hit mesh vertex index for one camera view."""
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic.astype(np.float64),
        extrinsic_matrix=pose_C_W.astype(np.float64),
        width_px=width,
        height_px=height,
    )
    ans = scene.cast_rays(rays)
    hit_triangle_ids = ans["primitive_ids"].numpy()
    hit_mask = hit_triangle_ids < mesh_triangles.shape[0]
    hit_vertex_idx = mesh_triangles[hit_triangle_ids[hit_mask]][:, 0]
    return hit_mask, hit_vertex_idx


def patch_majority_vote(obj_id_map: np.ndarray, patch_w: int, patch_h: int, th: float):
    """Winning object id per patch, 0 where no id covers more than th of the patch."""
    image_h, image_w = obj_id_map.shape
    patch_h_size = int(image_h / patch_h)
    patch_w_size = int(image_w / patch_w)
    patch_annos = np.zeros((patch_h, patch_w), dtype=np.uint64)
    for i in range(patch_h):
        h_start, h_end = round(i * patch_h_size), round((i + 1) * patch_h_size)
        for j in range(patch_w):
            w_start, w_end = round(j * patch_w_size), round((j + 1) * patch_w_size)
            patch_size = (w_end - w_start) * (h_end - h_start)
            anno = obj_id_map[h_start:h_end, w_start:w_end]
            obj_ids, counts = np.unique(anno.reshape(-1), return_counts=True)
            max_idx = np.argmax(counts)
            if counts[max_idx] > th * patch_size:
                patch_annos[i, j] = obj_ids[max_idx]
    return patch_annos


def generate_scan(scans_dir: str, scan_id: str, cfg):
    mesh = o3d.io.read_triangle_mesh(osp.join(scans_dir, scan_id, scan_id + MESH_NAME))
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_triangles = np.asarray(mesh.triangles)

    sgfusion = PlyData.read(osp.join(scans_dir, scan_id, SGFUSION_PLY))["vertex"]
    sgfusion_points = np.stack([sgfusion["x"], sgfusion["y"], sgfusion["z"]], axis=1)
    sgfusion_labels = np.asarray(sgfusion["label"])

    # nearest-neighbor label transfer from the predicted cloud to the mesh vertices
    kdtree = o3c.nns.NearestNeighborSearch(o3c.Tensor(sgfusion_points, dtype=o3c.float32))
    kdtree.knn_index()
    idx, dist = kdtree.knn_search(o3c.Tensor(mesh_vertices, dtype=o3c.float32), 1)
    idx = idx.numpy().reshape(-1)
    valid = dist.numpy().reshape(-1) < LABEL_TRANSFER_DIST_TH**2
    mesh_labels = np.zeros(mesh_vertices.shape[0], dtype=np.int32)
    mesh_labels[valid] = sgfusion_labels[idx[valid]]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    intrinsic = load_frame_intrinsics(scans_dir, scan_id).copy()
    intrinsic[0, :] *= ANNO_RENDER_W / NATIVE_W
    intrinsic[1, :] *= ANNO_RENDER_H / NATIVE_H

    patch_annos = {}
    for frame_idx in load_frame_idxs(scans_dir, scan_id, skip=cfg.data.img.img_step):
        pose_W_C = np.loadtxt(osp.join(scans_dir, scan_id, "pose", f"{frame_idx}.txt"))
        if not np.isfinite(pose_W_C).all():
            continue
        hit_mask, hit_vertex_idx = raycast_frame(
            scene, mesh_triangles, intrinsic, np.linalg.inv(pose_W_C), ANNO_RENDER_W, ANNO_RENDER_H
        )
        obj_id_map = np.zeros((ANNO_RENDER_H, ANNO_RENDER_W), dtype=np.int32)
        obj_id_map[hit_mask] = mesh_labels[hit_vertex_idx]
        patch_annos[frame_idx] = patch_majority_vote(
            obj_id_map, cfg.data.img_encoding.patch_w, cfg.data.img_encoding.patch_h, PATCH_VOTE_TH
        )
    return patch_annos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); provides img_step, patch grid and the output name",
    )
    parser.add_argument("--scans-dir", required=True, help="directory with the ScanNet scans")
    parser.add_argument("--files-dir", required=True, help="the dataset files/ directory")
    parser.add_argument("--scene-list", required=True, help="txt file, one query scan per line")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    out_dir = osp.join(args.files_dir, cfg.data.gt_patch)
    os.makedirs(out_dir, exist_ok=True)
    scan_ids = [ln.strip() for ln in open(args.scene_list) if ln.strip()]
    for scan_id in tqdm(scan_ids):
        out_file = osp.join(out_dir, f"{scan_id}.pkl")
        if osp.isfile(out_file):
            continue
        annos = generate_scan(args.scans_dir, scan_id, cfg)
        with open(out_file, "wb") as f:
            pickle.dump(annos, f)


if __name__ == "__main__":
    main()
