"""
Median-split BVH over a triangle mesh, flattened into flat arrays for the CUDA raycaster.

build_bvh splits triangles by the median centroid until a leaf is small enough. flatten_bvh
emits the node and triangle arrays plus the leaf-order to original-order permutation used
to look up each hit triangle's UVs.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["BVHNode", "build_bvh", "flatten_bvh", "NODE_NUM_COLUMNS"]

# Flattened node layout (one row per node):
#   [bounds_min(3), bounds_max(3), num_triangles, start_triangle, left_idx, right_idx]
NODE_NUM_COLUMNS = 10
LEFT_CHILD_COLUMN = 8
RIGHT_CHILD_COLUMN = 9


class BVHNode:
    """A node of the BVH. Leaf nodes carry triangles, internal nodes carry children."""

    __slots__ = ("bounds", "left", "right", "triangles", "triangle_orig_indices")

    def __init__(
        self, bounds=None, left=None, right=None, triangles=None, triangle_orig_indices=None
    ):
        self.bounds = bounds  # (min_bounds[3], max_bounds[3])
        self.left = left  # BVHNode | None
        self.right = right  # BVHNode | None
        self.triangles = triangles  # (k, 3) leaf triangles
        self.triangle_orig_indices = triangle_orig_indices  # (k,) original indices, leaf only


def _compute_bounds(triangles: np.ndarray, vertices: np.ndarray):
    tv = vertices[triangles.flatten()]
    return np.min(tv, axis=0), np.max(tv, axis=0)


def build_bvh(
    triangles: np.ndarray,
    vertices: np.ndarray,
    max_triangles_per_leaf: int,
    triangle_orig_indices: np.ndarray | None = None,
) -> BVHNode:
    """Recursively build a median-split BVH and return its root node."""
    if triangle_orig_indices is None:
        triangle_orig_indices = np.arange(len(triangles))

    if len(triangles) <= max_triangles_per_leaf:
        return BVHNode(
            bounds=_compute_bounds(triangles, vertices),
            triangles=triangles,
            triangle_orig_indices=triangle_orig_indices,
        )

    min_b, max_b = _compute_bounds(triangles, vertices)
    axis = int(np.argmax(max_b - min_b))  # largest spread axis
    centroids = np.mean(vertices[triangles], axis=1)
    median = np.median(centroids[:, axis])  # balanced split
    left_mask = centroids[:, axis] <= median
    right_mask = ~left_mask

    left_t, left_o = triangles[left_mask], triangle_orig_indices[left_mask]
    right_t, right_o = triangles[right_mask], triangle_orig_indices[right_mask]

    # Degenerate split (all centroids on one side): make this a leaf.
    if len(left_t) == 0 or len(right_t) == 0:
        return BVHNode(
            bounds=(min_b, max_b), triangles=triangles, triangle_orig_indices=triangle_orig_indices
        )

    left = build_bvh(left_t, vertices, max_triangles_per_leaf, left_o)
    right = build_bvh(right_t, vertices, max_triangles_per_leaf, right_o)
    bounds_min = np.minimum(left.bounds[0], right.bounds[0])
    bounds_max = np.maximum(left.bounds[1], right.bounds[1])
    return BVHNode(bounds=(bounds_min, bounds_max), left=left, right=right)


def flatten_bvh(node: BVHNode, nodes: list, triangles: list, triangle_orig_indices: list) -> int:
    """Flatten the tree into flat lists appending in place and return this node's row index."""
    node_index = len(nodes)
    is_leaf = node.triangles is not None
    nodes.append(
        [
            *node.bounds[0],
            *node.bounds[1],
            len(node.triangles) if is_leaf else 0,
            len(triangles) if is_leaf else -1,
            -1,  # left child row (filled below for internal nodes)
            -1,  # right child row
        ]
    )

    if not is_leaf:
        nodes[node_index][LEFT_CHILD_COLUMN] = flatten_bvh(
            node.left, nodes, triangles, triangle_orig_indices
        )
        nodes[node_index][RIGHT_CHILD_COLUMN] = flatten_bvh(
            node.right, nodes, triangles, triangle_orig_indices
        )
    else:
        triangles.extend(node.triangles)
        triangle_orig_indices.extend(node.triangle_orig_indices)

    return node_index


def mean_edge_length(vertices: np.ndarray, triangles: np.ndarray) -> float:
    v, t = vertices, triangles
    return float(
        np.mean(
            np.concatenate(
                [
                    np.linalg.norm(v[t[:, 0]] - v[t[:, 1]], axis=1),
                    np.linalg.norm(v[t[:, 1]] - v[t[:, 2]], axis=1),
                    np.linalg.norm(v[t[:, 2]] - v[t[:, 0]], axis=1),
                ]
            )
        )
    )


def transfer_obj_ids(
    decimated_vertices: np.ndarray,
    annotated_vertices: np.ndarray,
    annotated_triangles: np.ndarray,
    annotated_obj_ids: np.ndarray,
    k: int = 3,
    radius_factor: float = 1.5,
    radius: float | None = None,
) -> np.ndarray:
    """Assign each decimated vertex an object id via a radius-limited k-NN inverse-distance vote."""
    if radius is None:
        radius = radius_factor * mean_edge_length(annotated_vertices, annotated_triangles)

    tree = cKDTree(annotated_vertices)
    dists, idxs = tree.query(decimated_vertices, k=k, distance_upper_bound=radius, workers=-1)
    _, nearest = tree.query(
        decimated_vertices, k=1, workers=-1
    )  # fallback when nothing is in range

    n_ann = annotated_vertices.shape[0]
    out = np.empty(decimated_vertices.shape[0], dtype=np.int32)
    for i in range(decimated_vertices.shape[0]):
        neigh_idx = np.atleast_1d(idxs[i])
        neigh_dist = np.atleast_1d(dists[i])
        valid = (neigh_idx >= 0) & (neigh_idx < n_ann) & np.isfinite(neigh_dist)
        if not np.any(valid):
            out[i] = annotated_obj_ids[nearest[i]]
            continue
        sel_idx, sel_dist = neigh_idx[valid], neigh_dist[valid]
        if sel_idx.size == 1:
            out[i] = annotated_obj_ids[sel_idx[0]]
            continue
        weights = 1.0 / np.maximum(sel_dist, 1e-12)
        weight_per_label: dict = {}
        for label, w in zip(annotated_obj_ids[sel_idx], weights):
            weight_per_label[label] = weight_per_label.get(label, 0.0) + w
        out[i] = max(weight_per_label.items(), key=lambda kv: kv[1])[0]
    return out
