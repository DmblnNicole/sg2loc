"""
Precomputed BVH geometry for a scene: load it from disk and hold both CPU and GPU copies.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

import numpy as np
from numba import cuda


@dataclass
class SceneGeometry:
    """Geometry for a scene, loaded once and shared by every query sequence that localizes into it."""

    vertices: np.ndarray  # CPU
    obj_ids: np.ndarray  # CPU, per-vertex object ids
    vertices_gpu: object  # the *_gpu fields are GPU device arrays for the raycaster kernels
    bvh_nodes_gpu: object
    bvh_triangles_gpu: object
    uv_coords_gpu: object
    uv_indices_gpu: object
    triangle_orig_indices_gpu: object
    # per-vertex RGB uint8, absent for scenes with a UV texture which render through uv_coords
    vertex_colors_gpu: object = None


def load_scene_geometry(bvh_dir: str) -> SceneGeometry:
    """Load a scene's precomputed BVH .npy files and upload the geometry to the GPU."""
    vertices = np.load(osp.join(bvh_dir, "vertices.npy"))
    flat_nodes = np.load(osp.join(bvh_dir, "bvh_nodes.npy"))
    flat_triangles = np.load(osp.join(bvh_dir, "bvh_triangles.npy"))
    obj_ids = np.load(osp.join(bvh_dir, "obj_ids.npy"))
    uv_coords = np.load(osp.join(bvh_dir, "uv_coords.npy"))
    uv_indices = np.load(osp.join(bvh_dir, "uv_indices.npy"))
    triangle_orig_indices = np.load(osp.join(bvh_dir, "triangle_orig_indices.npy"))
    vertex_colors_gpu = None
    vertex_colors_path = osp.join(bvh_dir, "vertex_colors.npy")
    if osp.exists(vertex_colors_path):
        vertex_colors = np.load(vertex_colors_path)
        vertex_colors_gpu = cuda.to_device(np.ascontiguousarray(vertex_colors))
        # the canonical uv mapping makes the kernel's interpolated uv the hit's barycentric (u, v)
        uv_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        uv_indices = np.tile(
            np.array([0, 1, 2], dtype=np.int64), (int(triangle_orig_indices.max()) + 1, 1)
        )
    return SceneGeometry(
        vertices=vertices,
        obj_ids=obj_ids,
        vertices_gpu=cuda.to_device(vertices),
        bvh_nodes_gpu=cuda.to_device(flat_nodes),
        bvh_triangles_gpu=cuda.to_device(flat_triangles),
        uv_coords_gpu=cuda.to_device(uv_coords),
        uv_indices_gpu=cuda.to_device(uv_indices),
        triangle_orig_indices_gpu=cuda.to_device(triangle_orig_indices),
        vertex_colors_gpu=vertex_colors_gpu,
    )
