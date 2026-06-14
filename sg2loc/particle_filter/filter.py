"""
Particle filter that samples, propagates and resamples pose particles.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from sg2loc.particle_filter.observation import ObservationModel
from sg2loc.particle_filter.raycasting import RayCasting
from sg2loc.particle_filter.scene import SceneGeometry, load_scene_geometry
from sg2loc.utils import torch_util

logger = logging.getLogger(__name__)


class ParticleFilter:
    """Pose particle filter, subclassed per dataset to provide the data access."""

    def __init__(
        self,
        cfg: Any,
        data_dict: dict,
        patch_obj_sim_T: list,
        scene: SceneGeometry | None = None,
    ) -> None:
        setup_t0 = time.time()
        self.cfg = cfg
        self.data_dict = data_dict
        self.patch_obj_sim_T = patch_obj_sim_T  # (B, P_H, P_W)
        self.sequence_length = cfg.particle_filter.sequence_length
        self.scans_scenes_dir = cfg.particle_filter.scans_scenes_dir

        self.num_pass = 1
        self.debug_snapshots: list = []

        self.scan_id = data_dict["scan_ids"][0]  # scan
        self.target_scan_id = data_dict["scan_ids_temp"][0]  # rescan
        # BVH geometry for the target scene, cached by the runner, loaded here when standalone
        if scene is None:
            bvh_dir = osp.join(self.cfg.particle_filter.preprocess.output_dir, self.target_scan_id)
            scene = load_scene_geometry(bvh_dir)
        self.scene = scene
        frame_idxs = self.data_dict["frame_idxs"]
        query_image_poses_scan = self.load_query_poses(self.scan_id, frame_idxs)
        self.query_image_poses_scan = query_image_poses_scan
        # align the query poses into the map scan frame where the particles are sampled
        recan_to_scan = self.map_to_query_transform(self.target_scan_id)
        scan_to_rescan = np.linalg.inv(recan_to_scan)
        query_image_poses_rescan = np.einsum("ij,njk->nik", scan_to_rescan, query_image_poses_scan)
        self.query_image_poses = query_image_poses_rescan

        self.weights, self.poses = self.sample_initial_particles()

        obj_ids = self.data_dict["scene_graphs"]["obj_ids"]
        obj_ids = torch_util.release_cuda_torch(obj_ids)
        obj_counts = self.data_dict["scene_graphs"]["tot_obj_count"]
        self.obj_ids_slices = []
        start = 0
        for count in obj_counts:
            self.obj_ids_slices.append(obj_ids[start : start + count])
            start += count
        scene_graphs = self.data_dict["scene_graphs"]["scene_ids"]
        self.scene_graph_to_index = {scene_id[0]: idx for idx, scene_id in enumerate(scene_graphs)}

        self.intrinsics = self.load_intrinsics(self.scan_id)
        self.img_width = self.cfg.data.img.w
        self.img_height = self.cfg.data.img.h
        # constant Z-depth to Euclidean ray-distance factor for the query intrinsics
        us, vs = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
        x = (us - self.intrinsics[0, 2]) / self.intrinsics[0, 0]
        y = (vs - self.intrinsics[1, 2]) / self.intrinsics[1, 1]
        self.depth_ray_norm = np.sqrt(1.0 + x * x + y * y)
        self.setup_seconds = time.time() - setup_t0

    def sample_initial_particles(self) -> tuple:
        scan_id = self.target_scan_id
        vertices = self.scene.vertices
        obj_ids = self.scene.obj_ids
        floor_mask = self.floor_mask(vertices, obj_ids)
        floor_vertices = vertices[floor_mask]

        if len(floor_vertices) == 0:
            raise ValueError(
                f"no floor vertices in scene {scan_id}, cannot sample initial particles"
            )

        floor_mean = np.mean(floor_vertices, axis=0)
        centered_floor = floor_vertices - floor_mean
        # full_matrices=False skips the huge (N, N) U matrix, only Vt is used
        _, _, Vt = np.linalg.svd(centered_floor, full_matrices=False)
        floor_normal = Vt[2, :]

        if floor_normal[2] < 0:
            floor_normal = -floor_normal

        floor_height = np.dot(floor_mean, floor_normal) / floor_normal[2]
        projected_floor_vertices = floor_vertices - np.outer(
            (floor_vertices - floor_mean) @ floor_normal, floor_normal
        )

        bbox_min = np.min(projected_floor_vertices, axis=0)
        bbox_max = np.max(projected_floor_vertices, axis=0)
        max_scene_height = np.max(vertices[:, 2])

        bbox_min = np.array([bbox_min[0], bbox_min[1], floor_height])
        bbox_max = np.array([bbox_max[0], bbox_max[1], max_scene_height])

        grid_resolution = self.cfg.particle_filter.sampling.grid_resolution
        x_coords = np.arange(bbox_min[0], bbox_max[0], grid_resolution)
        y_coords = np.arange(bbox_min[1], bbox_max[1], grid_resolution)

        grid_points = []
        for height in self.cfg.particle_filter.sampling.heights:
            absolute_height = floor_height + height
            grid_points.append(
                np.array(np.meshgrid(x_coords, y_coords, [absolute_height])).T.reshape(-1, 3)
            )

        grid_points = np.vstack(grid_points)

        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

        half_cell = np.array([grid_resolution / 2, grid_resolution / 2, 0])
        samples_per_cell = self.cfg.particle_filter.sampling.samples_per_cell
        random_translations = np.random.uniform(
            (grid_points - half_cell)[:, None, :],
            (grid_points + half_cell)[:, None, :],
            size=(len(grid_points), samples_per_cell, 3),
        ).reshape(-1, 3)

        axis_col = self.gravity_axis_col
        free_col = 1 - axis_col
        fixed_axis = self.query_image_poses[0][:3, axis_col]
        fixed_axis = fixed_axis / np.linalg.norm(fixed_axis)

        seed = np.zeros(3)
        seed[free_col] = 1.0
        free_axis = seed - np.dot(seed, fixed_axis) * fixed_axis
        free_axis = free_axis / np.linalg.norm(free_axis)

        columns = [None, None, None]
        columns[axis_col] = fixed_axis
        columns[free_col] = free_axis
        columns[2] = np.cross(columns[0], columns[1])
        columns[2] = columns[2] / np.linalg.norm(columns[2])

        fixed_rotation = np.column_stack(columns)
        # kept for drawing fresh global particles during resampling
        self.fixed_gravity_axis = fixed_axis
        self.fixed_rotation = fixed_rotation

        random_roll_angles = np.random.uniform(
            self.cfg.particle_filter.sampling.roll_min,
            self.cfg.particle_filter.sampling.roll_max,
            size=(len(random_translations),),
        )

        rotation_matrices = (
            R.from_rotvec(np.outer(random_roll_angles, fixed_axis)).as_matrix() @ fixed_rotation
        )

        poses = np.eye(4).reshape(1, 4, 4).repeat(len(random_translations), axis=0)
        poses[:, :3, :3] = rotation_matrices
        poses[:, :3, 3] = random_translations

        self.num_particles = len(random_translations)

        weights = np.full(len(random_translations), 1.0 / self.num_particles)
        return weights, poses

    def calc_rel_pose(self, world_pose_cam1: np.ndarray, world_pose_cam2: np.ndarray) -> tuple:
        # relative pose of cam2 in the coordinate system of cam1, from cam-to-world poses
        cam2_pose_cam1 = np.linalg.inv(world_pose_cam1) @ world_pose_cam2
        cam2_rot_cam1 = cam2_pose_cam1[:3, :3]
        cam2_trans_cam1 = cam2_pose_cam1[:3, 3]
        return cam2_rot_cam1, cam2_trans_cam1

    def transition_model(self, query_image: int) -> None:
        world_pose_cam1 = self.query_image_poses[query_image - 1]
        world_pose_cam2 = self.query_image_poses[query_image]
        cam2_rot_cam1, cam2_trans_cam1 = self.calc_rel_pose(world_pose_cam1, world_pose_cam2)

        poses = self.poses

        t_mean = self.cfg.particle_filter.transition_model.translation_noise_mean
        t_std = self.cfg.particle_filter.transition_model.translation_noise_stddev
        r_mean = self.cfg.particle_filter.transition_model.rotation_noise_mean
        r_std = self.cfg.particle_filter.transition_model.rotation_noise_stddev

        noisy_translations = cam2_trans_cam1 + np.random.normal(
            t_mean, t_std, size=(self.num_particles, 3)
        )

        delta_phi = np.random.normal(r_mean, r_std, size=self.num_particles)
        delta_psi = np.random.normal(r_mean, r_std, size=self.num_particles)
        delta_theta = np.random.normal(r_mean, r_std, size=self.num_particles)

        R_x = R.from_rotvec(np.outer(delta_phi, [1, 0, 0])).as_matrix()
        R_y = R.from_rotvec(np.outer(delta_theta, [0, 1, 0])).as_matrix()
        R_z = R.from_rotvec(np.outer(delta_psi, [0, 0, 1])).as_matrix()
        noisy_rotations = R_z @ R_y @ R_x @ cam2_rot_cam1

        current_translations = poses[:, :3, 3]
        current_rotations = poses[:, :3, :3]

        new_translations = current_translations + np.einsum(
            "nij,nj->ni", current_rotations, noisy_translations
        )  # rotate the noisy translations into the world frame

        new_rotations = np.einsum("nij,njk->nik", current_rotations, noisy_rotations)

        axis_col = self.gravity_axis_col
        free_col = 1 - axis_col
        known_axis = self.query_image_poses[query_image][:3, axis_col]
        known_axis = known_axis / np.linalg.norm(known_axis)
        known_axis = np.tile(known_axis, (self.num_particles, 1))
        new_rotations[:, :, axis_col] = known_axis  # enforce the known gravity axis
        new_rotations[:, :, free_col] -= (
            np.sum(new_rotations[:, :, free_col] * known_axis, axis=-1, keepdims=True) * known_axis
        )  # orthogonalize the free axis
        new_rotations[:, :, free_col] /= np.linalg.norm(
            new_rotations[:, :, free_col], axis=-1, keepdims=True
        )
        new_rotations[:, :, 2] = np.cross(new_rotations[:, :, 0], new_rotations[:, :, 1])
        new_rotations[:, :, 2] /= np.linalg.norm(new_rotations[:, :, 2], axis=-1, keepdims=True)

        updated_poses = np.eye(4).reshape(1, 4, 4).repeat(self.num_particles, axis=0)
        updated_poses[:, :3, :3] = new_rotations
        updated_poses[:, :3, 3] = new_translations
        self.poses = updated_poses

    def sample_random_particles(self, count: int) -> np.ndarray:
        """Draw count fresh global particles from the initial sampling distribution."""
        xy = np.random.uniform(self.bbox_min[:2], self.bbox_max[:2], size=(count, 2))
        heights = np.array(self.cfg.particle_filter.sampling.heights)
        z = self.bbox_min[2] + np.random.choice(heights, size=count)
        rolls = np.random.uniform(
            self.cfg.particle_filter.sampling.roll_min,
            self.cfg.particle_filter.sampling.roll_max,
            size=count,
        )
        rotations = (
            R.from_rotvec(np.outer(rolls, self.fixed_gravity_axis)).as_matrix()
            @ self.fixed_rotation
        )
        poses = np.eye(4).reshape(1, 4, 4).repeat(count, axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = np.column_stack([xy, z])
        return poses

    def stratified_resampling(self, weights: np.ndarray, poses: np.ndarray, N: int) -> tuple:
        # Adapted from https://github.com/jelfring/particle-filter-tutorial/blob/master/core/resampling/resampler.py
        Q = np.cumsum(weights)

        # each 1/N interval contributes exactly one sample
        u0s = np.random.uniform(1e-10, 1.0 / N, N)
        us = np.minimum(u0s + np.arange(N) / N, Q[-1])  # clamp, Q[-1] is not always exactly 1.0
        selected = np.searchsorted(Q, us, side="left")
        return np.full(N, 1.0 / N), poses[selected]

    def propagate_pose_to_first_image(self, top_pose: np.ndarray, from_idx: int = -1) -> np.ndarray:
        """Move the pose from the source frame to frame 0 with the known relative transform."""
        rel_rot, rel_trans = self.calc_rel_pose(
            self.query_image_poses[from_idx], self.query_image_poses[0]
        )

        current_rot = top_pose[:3, :3]
        current_trans = top_pose[:3, 3]

        new_trans = current_trans + np.dot(current_rot, rel_trans)
        new_rot = np.dot(current_rot, rel_rot)

        updated_pose = np.eye(4)
        updated_pose[:3, :3] = new_rot
        updated_pose[:3, 3] = new_trans

        return updated_pose

    def _reinit_widths(self) -> tuple:
        if self.num_pass == 1:
            return (
                self.cfg.particle_filter.first_reinit_translation_halfwidth,
                self.cfg.particle_filter.first_reinit_rotation_std,
            )
        if self.num_pass == 2:
            return (
                self.cfg.particle_filter.second_reinit_translation_halfwidth,
                self.cfg.particle_filter.second_reinit_rotation_std,
            )
        raise ValueError("Invalid num_pass value")

    def _sample_around_pose(
        self, pose: np.ndarray, count: int, translation_halfwidth: float, rotation_std: float
    ) -> np.ndarray:
        top_translation = pose[:3, 3]

        reinit_bbox_min = np.maximum(top_translation - translation_halfwidth, self.bbox_min)
        reinit_bbox_max = np.minimum(top_translation + translation_halfwidth, self.bbox_max)

        sampled_translations = np.random.uniform(reinit_bbox_min, reinit_bbox_max, size=(count, 3))

        axis_col = self.gravity_axis_col
        free_col = 1 - axis_col
        known_axis = self.query_image_poses[0][:3, axis_col]
        known_axis = known_axis / np.linalg.norm(known_axis)

        random_roll_angles = np.random.normal(0, rotation_std, count)
        local_axis = np.zeros(3)
        local_axis[axis_col] = 1.0
        R_axis_batch = R.from_rotvec(np.outer(random_roll_angles, local_axis)).as_matrix()

        # orthonormal base around the known gravity axis (axes are columns)
        base_rotation = pose[:3, :3].copy()
        base_rotation[:, axis_col] = known_axis
        base_rotation[:, axis_col] /= np.linalg.norm(base_rotation[:, axis_col])
        base_rotation[:, free_col] -= np.dot(base_rotation[:, free_col], known_axis) * known_axis
        base_rotation[:, free_col] /= np.linalg.norm(base_rotation[:, free_col])
        base_rotation[:, 2] = np.cross(base_rotation[:, 0], base_rotation[:, 1])
        rotations = base_rotation @ R_axis_batch

        poses = np.eye(4).reshape(1, 4, 4).repeat(count, axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = sampled_translations
        return poses

    def reinitialize_particles_around_pose(self, pose: np.ndarray) -> tuple:
        translation_halfwidth, rotation_std = self._reinit_widths()
        num_particles = self.num_particles
        poses = self._sample_around_pose(pose, num_particles, translation_halfwidth, rotation_std)
        weights = np.full(num_particles, 1.0 / num_particles)
        self.num_particles = num_particles
        return weights, poses

    def select_top_modes(self) -> list:
        """Greedily extract up to reinit_num_modes (pose, mass) pairs from the particle set."""
        pf_cfg = self.cfg.particle_filter
        k = pf_cfg.reinit_num_modes
        if k <= 1:
            return [(self.get_maximum_likelihood_estimate(), 1.0)]
        translations = self.poses[:, :3, 3]
        yaws = self.particle_yaws(self.poses)
        remaining = self.weights.copy()
        modes = []
        for _ in range(k):
            best = int(np.argmax(remaining))
            if remaining[best] <= 0.0:
                break
            dist = np.linalg.norm(translations - translations[best], axis=1)
            yaw_diff = np.abs((yaws - yaws[best] + np.pi) % (2 * np.pi) - np.pi)
            member = (dist <= pf_cfg.reinit_mode_separation) & (
                yaw_diff <= pf_cfg.reinit_mode_yaw_separation
            )
            modes.append((self.poses[best].copy(), float(remaining[member].sum())))
            remaining[member] = 0.0
        total = sum(mass for _, mass in modes)
        return [(pose, mass / total) for pose, mass in modes]

    def accumulate_modes(self, image: int) -> None:
        """Get the current top modes, brought to frame 0, and merge them into the pass-wide list."""
        pf_cfg = self.cfg.particle_filter
        for pose, mass in self.select_top_modes():
            pose0 = self.propagate_pose_to_first_image(pose, from_idx=image)
            yaw0 = self.particle_yaws(pose0[None])[0]
            for entry in self.accumulated_modes:
                dist = np.linalg.norm(entry[0][:3, 3] - pose0[:3, 3])
                yaw_diff = abs((entry[1] - yaw0 + np.pi) % (2 * np.pi) - np.pi)
                if (
                    dist <= pf_cfg.reinit_mode_separation
                    and yaw_diff <= pf_cfg.reinit_mode_yaw_separation
                ):
                    # a mode seen at several frames keeps its best mass
                    if mass > entry[2]:
                        entry[0], entry[1], entry[2] = pose0, yaw0, mass
                    break
            else:
                self.accumulated_modes.append([pose0, yaw0, mass])

    def reinitialize_particles_around_modes(self, modes: list) -> tuple:
        """Reseed the particle set around the top pass modes with counts proportional to mass."""
        translation_halfwidth, rotation_std = self._reinit_widths()
        num_particles = self.num_particles
        # the floor keeps weak but distinct hypotheses alive into the next pass
        floor = num_particles // (4 * len(modes)) if len(modes) > 1 else 0
        counts = np.maximum(
            floor, np.round(np.array([mass for _, mass in modes]) * num_particles).astype(int)
        )
        counts[np.argmax(counts)] += num_particles - counts.sum()
        poses = np.concatenate(
            [
                self._sample_around_pose(pose, count, translation_halfwidth, rotation_std)
                for (pose, _), count in zip(modes, counts)
            ]
        )
        weights = np.full(num_particles, 1.0 / num_particles)
        self.num_particles = num_particles
        return weights, poses

    def compute_required_number_of_particles_kld(
        self, k_bins: int, epsilon: float, upper_quantile: float
    ) -> int:
        x = 1.0 - 2.0 / (9.0 * (k_bins - 1)) + np.sqrt(2.0 / (9.0 * (k_bins - 1))) * upper_quantile
        return int(np.ceil((k_bins - 1) / (2.0 * epsilon) * x**3))

    def load_query_sensor_depth(self, scan_id: str, frame_idx: int) -> np.ndarray:
        # sensor depth is low-resolution mm Z-depth, 0 marks invalid pixels
        path = self.sensor_depth_path(scan_id, frame_idx)
        depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = depth_mm.astype(np.float32) / 1000.0
        depth = cv2.resize(
            depth, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST
        )
        depth = depth * self.depth_ray_norm  # Z-depth to Euclidean, matching the raycaster
        return depth.astype(np.float32)

    def run_particle_filter(self) -> tuple:
        setup_t0 = time.time()
        depth_maps = [
            self.load_query_sensor_depth(self.scan_id, self.data_dict["frame_idxs"][i])
            for i in range(self.sequence_length)
        ]

        # coarse-to-fine passes share the uploaded BVH geometry, only the ray stride changes
        obs = self.cfg.particle_filter.observation_model
        pass_strides = [obs.first_stride, obs.second_stride, obs.third_stride]
        pass_strides = pass_strides[: self.cfg.particle_filter.num_passes]
        raycaster = RayCasting(self.cfg, self.intrinsics, pass_strides[0])
        raycaster.set_scene(self.scene)
        self.raycaster = raycaster

        # per-sequence setup time counts toward localization, recorded as a dedicated row
        self.setup_seconds += time.time() - setup_t0
        self.per_frame_data = [("setup", self.setup_seconds, self.num_particles)]

        pose = None
        self.frame_mle_poses: dict = {}
        for num_pass, stride in enumerate(pass_strides, start=1):
            self.num_pass = num_pass
            self.final_pass = num_pass == len(pass_strides)
            raycaster.stride = stride
            pose = self.run_pass(depth_maps)
            if pose is None:
                break
            if num_pass < len(pass_strides):
                # keep the top pass modes alive instead of committing to the single MLE
                if self.accumulated_modes:
                    # top modes seen at any frame of the pass, in frame-0 space
                    k = self.cfg.particle_filter.reinit_num_modes
                    top = sorted(self.accumulated_modes, key=lambda e: -e[2])[:k]
                    total = sum(mass for _, _, mass in top)
                    modes = [(pose, mass / total) for pose, _, mass in top]
                else:
                    modes = [
                        (self.propagate_pose_to_first_image(p), mass)
                        for p, mass in self.select_top_modes()
                    ]
                if num_pass == len(pass_strides) - 1:
                    # frame-0 estimate for the final pass (frame 0 gets no update in run_pass)
                    self.frame_mle_poses[self.data_dict["frame_idxs"][0]] = modes[0][0]
                self.weights, self.poses = self.reinitialize_particles_around_modes(modes)
                self._debug_dump("reinit", 0)

        if pose is None:
            logger.warning("no valid particles found")
            return None, None, None
        return pose, self.query_image_poses[-1], self.per_frame_data

    def run_pass(self, depth_maps: list) -> np.ndarray | None:
        """Run every sequence frame for the current pass and return the MLE pose after the last one."""
        observation_model = ObservationModel(self)
        required_particles = self.cfg.particle_filter.min_particles
        self.accumulated_modes = []
        accumulate = self.cfg.particle_filter.reinit_num_modes > 1 and not self.final_pass
        self._debug_dump("start", 0)
        for image in range(1, self.sequence_length):
            start_frame_time = time.time()
            self.transition_model(image)
            observation_model.update_weights(image, depth_maps[image])
            self._debug_dump("update", image)
            if accumulate:
                self.accumulate_modes(image)

            if self.final_pass:
                # per-frame pose estimate, taken before resampling flattens the weights
                self.frame_mle_poses[self.data_dict["frame_idxs"][image]] = (
                    self.get_maximum_likelihood_estimate()
                )

            particle_count = len(self.weights)

            if image < self.sequence_length - 1:
                self.current_image = image
                required_particles = self.kld_resample(required_particles)
                self._debug_dump("resample", image)

            self.per_frame_data.append(
                (
                    self.data_dict["frame_idxs"][image],
                    time.time() - start_frame_time,
                    particle_count,
                )
            )
        return self.get_maximum_likelihood_estimate()

    def particle_yaws(self, poses: np.ndarray) -> np.ndarray:
        """Compute the yaw of every particle around the gravity-aligned camera axis."""
        rot = poses[:, :3, :3]
        gravity = rot[:, :, self.gravity_axis_col]  # particle gravity axis
        ref = np.array(
            [1.0, 0.0, 0.0]
        )  # fixed world ref, safe since the gravity axis is near vertical
        ref_in_plane = ref - (gravity @ ref)[:, None] * gravity  # project ref off the gravity axis
        ref_in_plane = ref_in_plane / np.linalg.norm(ref_in_plane, axis=1, keepdims=True)
        bitangent = np.cross(gravity, ref_in_plane)
        free_axis = rot[:, :, 1 - self.gravity_axis_col]
        return np.arctan2(
            np.einsum("ni,ni->n", free_axis, bitangent),
            np.einsum("ni,ni->n", free_axis, ref_in_plane),
        )

    def kld_resample(self, required_particles: int) -> int:
        """KLD-adaptive resampling: bin the (x, y, z, yaw) states, update the count, and resample."""
        poses = self.poses
        yaw = self.particle_yaws(poses)
        translations = poses[:, :3, 3]
        resolutions = np.array(self.cfg.particle_filter.resolutions)
        xyz_bins = np.floor(translations / resolutions[:3]).astype(int)
        yaw_bins = np.floor(yaw / resolutions[3]).astype(int)
        bins_with_support = set(zip(xyz_bins[:, 0], xyz_bins[:, 1], xyz_bins[:, 2], yaw_bins))

        if len(bins_with_support) > 1:
            required_particles = self.compute_required_number_of_particles_kld(
                len(bins_with_support),
                self.cfg.particle_filter.epsilon,
                self.cfg.particle_filter.upper_quantile,
            )
        required_particles = max(self.cfg.particle_filter.min_particles, required_particles)
        required_particles = min(self.cfg.particle_filter.max_particles, required_particles)

        self.num_particles = required_particles
        self.weights, self.poses = self.stratified_resampling(
            self.weights, self.poses, required_particles
        )
        inject = self.cfg.particle_filter.resample_random_count
        if inject > 0 and self.current_image % self.cfg.particle_filter.resample_random_every == 0:
            # injected random particles let the filter escape a wrong dominant mode without displacing survivors
            self.poses = np.concatenate([self.poses, self.sample_random_particles(inject)])
            self.weights = np.full(len(self.poses), 1.0 / len(self.poses))
            self.num_particles = len(self.poses)
        return required_particles

    def get_maximum_likelihood_estimate(self) -> np.ndarray:
        return self.poses[np.argmax(self.weights)]

    def _debug_dump(self, tag: str, image: int) -> None:
        """Record the particle state for the debug GIF when the run has --debug enabled."""
        if not os.environ.get("PF_DEBUG", ""):
            return
        anchor = str(self.data_dict["frame_idxs"][-1])
        only = os.environ.get("PF_DEBUG_ANCHORS", "")
        if only and int(anchor) not in [int(a) for a in only.split(",")]:
            return
        n = len(self.poses)
        sel = np.random.default_rng(0).choice(n, min(n, 5000), replace=False)
        gt = self.query_image_poses[image]
        mle = self.get_maximum_likelihood_estimate()
        self.debug_snapshots.append(
            {
                "xyz": self.poses[sel, :3, 3].astype(np.float32),
                "yaw": self.particle_yaws(self.poses[sel]).astype(np.float32),
                "weights": self.weights[sel].astype(np.float32),
                "gt": np.array([*gt[:3, 3], self.particle_yaws(gt[None])[0]], dtype=np.float32),
                "mle": np.array(
                    [*mle[:3, 3], self.particle_yaws(mle[None])[0]], dtype=np.float32
                ),
                "num_pass": self.num_pass,
                "frame_id": self.data_dict["frame_idxs"][image],
                "tag": tag,
            }
        )
