"""
CUDA BVH raycaster: ray generation, traversal, triangle tests and the semantic scoring kernels.
"""

import math

import numpy as np
from numba import cuda, float32, int32

from sg2loc.particle_filter.scene import SceneGeometry


@cuda.jit(device=True)
def ray_box_intersect(ray_o, ray_d, box_min, box_max):
    """Check if a ray intersects an axis-aligned bounding box (AABB)."""
    tmin = -1e20  # smallest intersection distance
    tmax = 1e20  # largest intersection distance

    for i in range(3):  # check each axis
        if abs(ray_d[i]) > 1e-6:
            t1 = (box_min[i] - ray_o[i]) / ray_d[i]
            t2 = (box_max[i] - ray_o[i]) / ray_d[i]

            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        elif ray_o[i] < box_min[i] or ray_o[i] > box_max[i]:
            return False, tmin  # parallel ray outside the box

    return tmax >= tmin and tmax >= 0.0, tmin


@cuda.jit(device=True)
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Ray-triangle intersection (Moller-Trumbore)."""
    edge1 = cuda.local.array(3, dtype=float32)
    edge2 = cuda.local.array(3, dtype=float32)
    edge1[0] = v1[0] - v0[0]
    edge1[1] = v1[1] - v0[1]
    edge1[2] = v1[2] - v0[2]

    edge2[0] = v2[0] - v0[0]
    edge2[1] = v2[1] - v0[1]
    edge2[2] = v2[2] - v0[2]

    pvec = cuda.local.array(3, dtype=float32)
    pvec[0] = ray_d[1] * edge2[2] - ray_d[2] * edge2[1]
    pvec[1] = ray_d[2] * edge2[0] - ray_d[0] * edge2[2]
    pvec[2] = ray_d[0] * edge2[1] - ray_d[1] * edge2[0]

    det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2]
    if abs(det) < 1e-6:
        return False, 0.0, 0.0, 0.0  # parallel ray

    inv_det = 1.0 / det

    tvec = cuda.local.array(3, dtype=float32)
    tvec[0] = ray_o[0] - v0[0]
    tvec[1] = ray_o[1] - v0[1]
    tvec[2] = ray_o[2] - v0[2]

    u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det
    if u < 0.0 or u > 1.0:
        return False, 0.0, 0.0, 0.0  # the zeros are just dummy values

    qvec = cuda.local.array(3, dtype=float32)
    qvec[0] = tvec[1] * edge1[2] - tvec[2] * edge1[1]
    qvec[1] = tvec[2] * edge1[0] - tvec[0] * edge1[2]
    qvec[2] = tvec[0] * edge1[1] - tvec[1] * edge1[0]

    v = (ray_d[0] * qvec[0] + ray_d[1] * qvec[1] + ray_d[2] * qvec[2]) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False, 0.0, 0.0, 0.0

    # t is the intersection distance
    t = (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * inv_det
    if t < 1e-4:
        return False, 0.0, 0.0, 0.0

    return True, t, u, v


@cuda.jit(fastmath=True)
def ray_gen_and_traverse(
    K, Rs, tvecs, bvh_nodes, ray_leaf_info, img_width, img_height, stride, num_poses
):
    """BVH traversal: collect the leaf nodes each ray intersects, for triangle testing."""
    idx = cuda.grid(1)
    num_rays_per_pose = (img_width // stride) * (img_height // stride)
    total_rays = num_rays_per_pose * num_poses

    if idx >= total_rays:
        return

    # ray generation
    pose_idx = idx // num_rays_per_pose
    ray_idx = idx % num_rays_per_pose

    R = Rs[pose_idx]
    tvec = tvecs[pose_idx]

    i = (ray_idx // (img_width // stride)) * stride  # row index
    j = (ray_idx % (img_width // stride)) * stride  # column index

    # transform pixel coordinates to normalized image coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_camera = (j + 0.5 - cx) / fx
    y_camera = (i + 0.5 - cy) / fy
    z_camera = 1.0

    norm_factor = 1 / math.sqrt(x_camera**2 + y_camera**2 + z_camera**2)
    x_camera *= norm_factor
    y_camera *= norm_factor
    z_camera *= norm_factor

    # transform the ray direction to world coordinates
    ray_dir_world = cuda.local.array(3, dtype=float32)
    for k in range(3):
        ray_dir_world[k] = R[k, 0] * x_camera + R[k, 1] * y_camera + R[k, 2] * z_camera

    # BVH traversal
    stack = cuda.local.array(64, dtype=int32)
    stack_size = 0
    stack[stack_size] = 0  # start with the root node
    stack_size += 1

    leaf_count = 0
    # leaves go in columns 1..N, the last 7 columns hold the ray origin and direction
    max_leaf_count = ray_leaf_info.shape[1] - 8

    while stack_size > 0:
        stack_size -= 1
        node_idx = stack[stack_size]

        bounds_min = cuda.local.array(3, dtype=float32)
        bounds_max = cuda.local.array(3, dtype=float32)
        for k in range(3):
            bounds_min[k] = bvh_nodes[node_idx, k]
            bounds_max[k] = bvh_nodes[node_idx, k + 3]

        hit, _ = ray_box_intersect(tvec, ray_dir_world, bounds_min, bounds_max)
        if not hit:
            continue

        if bvh_nodes[node_idx, 6] > 0:  # leaf node
            if leaf_count < max_leaf_count:
                ray_leaf_info[idx, leaf_count + 1] = node_idx
                leaf_count += 1
        else:  # internal node
            left_child = int(bvh_nodes[node_idx, 8])
            right_child = int(bvh_nodes[node_idx, 9])
            stack[stack_size] = left_child
            stack_size += 1
            stack[stack_size] = right_child
            stack_size += 1

    # the number of intersected leaf nodes goes in the first slot
    ray_leaf_info[idx, 0] = leaf_count

    # store the ray origin and direction
    ray_leaf_info[idx, -7] = tvec[0]
    ray_leaf_info[idx, -6] = tvec[1]
    ray_leaf_info[idx, -5] = tvec[2]
    ray_leaf_info[idx, -4] = ray_dir_world[0]
    ray_leaf_info[idx, -3] = ray_dir_world[1]
    ray_leaf_info[idx, -2] = ray_dir_world[2]


@cuda.jit(fastmath=True)
def ray_triangle_intersect_in_leaf(
    ray_leaf_info,
    bvh_nodes,
    bvh_triangles,
    vertices,
    uv_coords,
    uv_indices,
    triangle_orig_indices,
    uv_hits,
    hit_record,
    depth_map,
    num_rays,
):
    idx = cuda.grid(1)
    if idx >= num_rays:
        return

    num_leaf_nodes = int(ray_leaf_info[idx, 0])  # num of intersected leaf nodes
    if num_leaf_nodes == 0:
        # ray hit no BVH leaf: write defined no-hit values, else uninitialized memory corrupts scores
        hit_record[idx] = -1
        depth_map[idx] = 1000
        uv_hits[idx, 0] = -1.0
        uv_hits[idx, 1] = -1.0
        return

    ray_o = cuda.local.array(3, dtype=float32)
    ray_d = cuda.local.array(3, dtype=float32)

    ray_o[0] = ray_leaf_info[idx, -7]
    ray_o[1] = ray_leaf_info[idx, -6]
    ray_o[2] = ray_leaf_info[idx, -5]

    ray_d[0] = ray_leaf_info[idx, -4]
    ray_d[1] = ray_leaf_info[idx, -3]
    ray_d[2] = ray_leaf_info[idx, -2]

    closest_t = 1e20
    hit_triangle_id = -1

    for leaf_idx in range(num_leaf_nodes):
        node_idx = int(ray_leaf_info[idx, leaf_idx + 1])
        start_idx = int(bvh_nodes[node_idx, 7])
        num_triangles = int(bvh_nodes[node_idx, 6])

        for t_idx in range(start_idx, start_idx + num_triangles):
            v0 = cuda.local.array(3, dtype=float32)
            v1 = cuda.local.array(3, dtype=float32)
            v2 = cuda.local.array(3, dtype=float32)

            for k in range(3):
                v0[k] = vertices[bvh_triangles[t_idx, 0], k]
                v1[k] = vertices[bvh_triangles[t_idx, 1], k]
                v2[k] = vertices[bvh_triangles[t_idx, 2], k]

            hit, t, u, v = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)
            if hit and t < closest_t:
                closest_t = t
                hit_triangle_id = t_idx
                best_u = u
                best_v = v

    hit_record[idx] = hit_triangle_id
    depth_map[idx] = closest_t if closest_t < 1e20 else 1000  # dummy value will be clamped away
    if hit_triangle_id >= 0:
        original_tri_id = triangle_orig_indices[hit_triangle_id]

        uv0 = cuda.local.array(2, dtype=float32)
        uv1 = cuda.local.array(2, dtype=float32)
        uv2 = cuda.local.array(2, dtype=float32)

        for k in range(2):
            uv0[k] = uv_coords[uv_indices[original_tri_id, 0], k]
            uv1[k] = uv_coords[uv_indices[original_tri_id, 1], k]
            uv2[k] = uv_coords[uv_indices[original_tri_id, 2], k]

        w = 1.0 - best_u - best_v
        uv_hits[idx, 0] = w * uv0[0] + best_u * uv1[0] + best_v * uv2[0]
        uv_hits[idx, 1] = w * uv0[1] + best_u * uv1[1] + best_v * uv2[1]
    else:
        uv_hits[idx, 0] = -1.0
        uv_hits[idx, 1] = -1.0


@cuda.jit(fastmath=True)
def process_rays_kernel(
    hit_record, obj_ids, obj_id_map, img_width, img_height, num_rays, num_poses, bvh_triangles
):
    """Build the per-pixel object-id map from ray hits."""
    global_idx = cuda.grid(1)
    if global_idx >= num_rays * num_poses:
        return

    particle_idx = global_idx // num_rays
    ray_idx = global_idx % num_rays

    i = ray_idx // img_width
    j = ray_idx % img_width

    if i < img_height and j < img_width:
        hit_triangle_id = hit_record[global_idx]
        if hit_triangle_id >= 0:
            obj_id_map[particle_idx, i, j] = obj_ids[bvh_triangles[hit_triangle_id, 0]]
        else:
            obj_id_map[particle_idx, i, j] = -1


@cuda.jit(fastmath=True)
def process_patches_kernel(
    obj_id_map,
    patches_global,
    PATCH_H,
    PATCH_W,
    PATCH_SIZE,
    img_width,
    img_height,
    num_poses,
    num_obj_bins,
):
    """Reduce the object-id map to one dominant object id per patch."""
    global_idx = cuda.grid(1)
    num_patches = PATCH_H * PATCH_W
    particle_idx = global_idx // num_patches
    patch_idx = global_idx % num_patches

    if particle_idx >= num_poses:
        return

    patch_i = patch_idx // PATCH_W
    patch_j = patch_idx % PATCH_W

    # fixed size for numba, the host guarantees num_obj_bins covers the largest object id
    pixel_counts = cuda.local.array(4096, dtype=int32)
    for idx in range(num_obj_bins):
        pixel_counts[idx] = 0

    for pi in range(PATCH_SIZE):
        for pj in range(PATCH_SIZE):
            x = patch_i * (PATCH_SIZE) + pi
            y = patch_j * (PATCH_SIZE) + pj
            if x < img_height and y < img_width:
                pixel_val = obj_id_map[particle_idx, x, y]
                if pixel_val >= 0:
                    pixel_counts[pixel_val] += 1

    max_count = 0
    dominant_id = 0
    for idx in range(num_obj_bins):
        if pixel_counts[idx] > max_count:
            max_count = pixel_counts[idx]
            dominant_id = idx

    # patches with many different object ids get object id 0, like the ground-truth annotation
    if max_count < (PATCH_SIZE * PATCH_SIZE) / 2:
        dominant_id = 0

    patches_global[particle_idx, patch_idx] = dominant_id


@cuda.jit
def compute_similarity_kernel(
    patches_global, predicted_node_ids_flatten, sim, sim_scores, PATCH_H, PATCH_W, num_poses
):
    """Accumulate the per-particle semantic similarity over agreeing patches."""
    global_idx = cuda.grid(1)
    if global_idx >= num_poses:
        return

    particle_idx = global_idx
    sim_score = 0.0

    for patch_idx in range(PATCH_H * PATCH_W):
        if patches_global[particle_idx, patch_idx] == predicted_node_ids_flatten[patch_idx]:
            sim_score += sim[patch_idx]

    sim_scores[particle_idx] = sim_score / (PATCH_H * PATCH_W)


class RayCasting:
    """Launches the raycast kernels for a scene at a configurable ray stride."""

    def __init__(self, cfg, intrinsics: np.ndarray, stride: int) -> None:
        self.cfg = cfg
        self.intrinsics = cuda.to_device(intrinsics)
        self.img_width, self.img_height = self.cfg.data.img.w, self.cfg.data.img.h
        self.patch_h, self.patch_w = (
            self.cfg.data.img_encoding.patch_h,
            self.cfg.data.img_encoding.patch_w,
        )
        self.stride = stride
        self.patch_h_size, self.patch_w_size = (
            self.img_height // self.patch_h,
            self.img_width // self.patch_w,
        )
        self.max_leaf_count = self.cfg.particle_filter.observation_model.max_leaf_count
        self.threads_per_block = self.cfg.particle_filter.observation_model.threads_per_block

    def set_scene(self, scene: SceneGeometry) -> None:
        """Point the raycaster at an already uploaded SceneGeometry."""
        self.num_obj_bins = int(scene.obj_ids.max()) + 1
        if self.num_obj_bins > 4096:
            raise ValueError(
                f"scene has object id {self.num_obj_bins - 1}, above the kernel histogram size 4096"
            )
        self.vertices = scene.vertices_gpu
        self.bvh_nodes_gpu = scene.bvh_nodes_gpu
        self.bvh_triangles_gpu = scene.bvh_triangles_gpu
        self.uv_coords = scene.uv_coords_gpu
        self.uv_indices = scene.uv_indices_gpu
        self.triangle_orig_indices = scene.triangle_orig_indices_gpu
        self.vertex_colors_gpu = scene.vertex_colors_gpu

    def to_device_frame_inputs(self, obj_ids, predicted_node_ids_flatten, sim) -> tuple:
        """Upload the per-frame constant scoring inputs once, for reuse across particle batches."""
        return (
            cuda.to_device(obj_ids.astype(np.int32)),
            cuda.to_device(predicted_node_ids_flatten.astype(np.int32)),
            cuda.to_device(sim.astype(np.float32)),
        )

    # used for scoring the particles
    def compute_rays_and_sim(
        self, Rs: np.ndarray, tvecs: np.ndarray, obj_ids_gpu, predicted_node_ids_gpu, sim_gpu
    ) -> tuple:
        num_poses = Rs.shape[0]
        num_rays = (self.img_width // self.stride) * (self.img_height // self.stride)
        total_rays = num_rays * num_poses

        patch_size = self.patch_h_size // self.stride

        hit_record = cuda.device_array(total_rays, dtype=np.int32)
        uv_hits = cuda.device_array((total_rays, 2), dtype=np.float32)
        depth_map = cuda.device_array(total_rays, dtype=np.float32)
        sim_scores = cuda.device_array(num_poses, dtype=np.float32)
        obj_id_map = cuda.device_array(
            (num_poses, self.img_height // self.stride, self.img_width // self.stride),
            dtype=np.int32,
        )
        patches_global = cuda.device_array((num_poses, self.patch_h * self.patch_w), dtype=np.int32)
        num_columns = 1 + self.max_leaf_count + 7
        ray_leaf_info = cuda.device_array((total_rays, num_columns), dtype=np.float32)
        Rs_gpu = cuda.to_device(Rs.astype(np.float32))
        tvecs_gpu = cuda.to_device(tvecs.astype(np.float32))

        blocks_for_rays = max(
            64 * 64, (total_rays + self.threads_per_block - 1) // self.threads_per_block
        )
        blocks_for_patches = max(
            64 * 64,
            (num_poses * self.patch_h * self.patch_w + self.threads_per_block - 1)
            // self.threads_per_block,
        )
        blocks_for_similarity = max(
            64 * 64, (num_poses + self.threads_per_block - 1) // self.threads_per_block
        )
        ray_gen_and_traverse[blocks_for_rays, self.threads_per_block](
            self.intrinsics,
            Rs_gpu,
            tvecs_gpu,
            self.bvh_nodes_gpu,
            ray_leaf_info,
            self.img_width,
            self.img_height,
            self.stride,
            num_poses,
        )
        ray_triangle_intersect_in_leaf[blocks_for_rays, self.threads_per_block](
            ray_leaf_info,
            self.bvh_nodes_gpu,
            self.bvh_triangles_gpu,
            self.vertices,
            self.uv_coords,
            self.uv_indices,
            self.triangle_orig_indices,
            uv_hits,
            hit_record,
            depth_map,
            total_rays,
        )
        process_rays_kernel[blocks_for_rays, self.threads_per_block](
            hit_record,
            obj_ids_gpu,
            obj_id_map,
            self.img_width // self.stride,
            self.img_height // self.stride,
            num_rays,
            num_poses,
            self.bvh_triangles_gpu,
        )
        process_patches_kernel[blocks_for_patches, self.threads_per_block](
            obj_id_map,
            patches_global,
            self.patch_h,
            self.patch_w,
            patch_size,
            self.img_width // self.stride,
            self.img_height // self.stride,
            num_poses,
            self.num_obj_bins,
        )
        compute_similarity_kernel[blocks_for_similarity, self.threads_per_block](
            patches_global,
            predicted_node_ids_gpu,
            sim_gpu,
            sim_scores,
            self.patch_h,
            self.patch_w,
            num_poses,
        )

        # the host copies below are synchronous, so torch can wrap the GPU uv_hits and hit_record zero-copy
        return (
            sim_scores.copy_to_host(),
            uv_hits,
            hit_record,
            depth_map.copy_to_host(),
        )

    # only used when rendering a whole image (and depth) is necessary, used by refiner and debug visualization
    def cast_depth_uv(self, Rs: np.ndarray, tvecs: np.ndarray) -> tuple:
        """Cast rays for the given poses and return per-ray uv hits, hit triangle ids and depths on the CPU."""
        num_poses = Rs.shape[0]
        num_rays = (self.img_width // self.stride) * (self.img_height // self.stride)
        total_rays = num_rays * num_poses
        hit_record = cuda.device_array(total_rays, dtype=np.int32)
        uv_hits = cuda.device_array((total_rays, 2), dtype=np.float32)
        depth_map = cuda.device_array(total_rays, dtype=np.float32)
        num_columns = 1 + self.max_leaf_count + 7
        ray_leaf_info = cuda.device_array((total_rays, num_columns), dtype=np.float32)
        Rs_gpu = cuda.to_device(Rs.astype(np.float32))
        tvecs_gpu = cuda.to_device(tvecs.astype(np.float32))
        blocks = max(64 * 64, (total_rays + self.threads_per_block - 1) // self.threads_per_block)
        ray_gen_and_traverse[blocks, self.threads_per_block](
            self.intrinsics,
            Rs_gpu,
            tvecs_gpu,
            self.bvh_nodes_gpu,
            ray_leaf_info,
            self.img_width,
            self.img_height,
            self.stride,
            num_poses,
        )
        ray_triangle_intersect_in_leaf[blocks, self.threads_per_block](
            ray_leaf_info,
            self.bvh_nodes_gpu,
            self.bvh_triangles_gpu,
            self.vertices,
            self.uv_coords,
            self.uv_indices,
            self.triangle_orig_indices,
            uv_hits,
            hit_record,
            depth_map,
            total_rays,
        )
        return uv_hits.copy_to_host(), hit_record.copy_to_host(), depth_map.copy_to_host()
