"""Per-particle observation model that scores poses via a CUDA BVH raycaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ObservationModel:
    """Scores particles via a CUDA BVH raycaster (semantic, RGB-SSIM and depth terms)."""

    def __init__(self, particle_filter: Any) -> None:
        self.pf = particle_filter
        cfg = particle_filter.cfg
        self.cfg = cfg
        self.patch_h, self.patch_w = cfg.data.img_encoding.patch_h, cfg.data.img_encoding.patch_w
        self.img_width, self.img_height = cfg.data.img.w, cfg.data.img.h
        obs = cfg.particle_filter.observation_model
        self.sim_noise_std = obs.sim_noise_stddev
        self.depth_error_cap_m = obs.depth_error_cap_m
        self.score_max_per_term = obs.score_max_per_term
        self.max_combined_sim = 3 * self.score_max_per_term  # 3 terms: semantic, depth, RGB
        self.device = torch.device("cuda")

    def calc_likelihood(self, sims: np.ndarray) -> np.ndarray:
        sims = np.clip(sims, 0, self.max_combined_sim)
        return np.exp(-((self.max_combined_sim - sims) ** 2) / (2 * (self.sim_noise_std**2)))

    _texture_cache = (None, None, 0, 0)  # (path, flat_uint8_gpu, tex_h, tex_w)

    def load_texture_gpu(self, path: str) -> tuple:
        # only for 3RScan we have texture
        cached_path, flat, th, tw = ObservationModel._texture_cache
        if cached_path != path:
            tex = np.array(Image.open(path).convert("RGB"))  # (th, tw, 3) uint8
            th, tw = tex.shape[:2]
            flat = torch.as_tensor(tex.reshape(-1, 3), device=self.device)  # uint8
            ObservationModel._texture_cache = (path, flat, th, tw)
        return flat, th, tw

    _query_cache = (None, None)  # (path, full-res GPU tensor before downsampling)

    def load_query_gpu(self, query_img_path: str, out_h: int, out_w: int) -> torch.Tensor:
        cached_path, q = ObservationModel._query_cache
        if cached_path != query_img_path:
            img = (
                Image.open(query_img_path)
                .convert("RGB")
                .resize((self.img_width, self.img_height), Image.BILINEAR)
            )
            q = torch.as_tensor(np.array(img), dtype=torch.float32, device=self.device)
            q = q.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, H, W)
            ObservationModel._query_cache = (query_img_path, q)
        return F.interpolate(q, size=(out_h, out_w), mode="bilinear", align_corners=False)

    # for 3RScan dataset
    def render_and_ssim_gpu(
        self,
        uv_hits,
        tex_flat: torch.Tensor,
        tex_h: int,
        tex_w: int,
        query: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        """Render each particle's mesh view from its uv hits and SSIM-score it against the query."""
        # uv_hits is a numba device array. as_tensor wraps it zero-copy (same dtype and device).
        uv = torch.as_tensor(uv_hits, dtype=torch.float32, device=self.device)  # (N, rays, 2)
        px_x = (uv[..., 0] * (tex_w - 1)).long().clamp_(0, tex_w - 1)
        px_y = ((1.0 - uv[..., 1]) * (tex_h - 1)).long().clamp_(0, tex_h - 1)  # flip v
        flat_idx = (px_y * tex_w + px_x).reshape(-1)
        n = uv.shape[0]
        rendered = (
            tex_flat[flat_idx].reshape(n, out_h, out_w, 3).permute(0, 3, 1, 2).float() / 255.0
        )
        return self.ssim_batch(rendered, query)

    # for ScanNet datset
    def render_barycentric_and_ssim_gpu(
        self,
        uv_hits,
        hit_record,
        vertex_colors: torch.Tensor,
        triangles: torch.Tensor,
        query: torch.Tensor,
        n_particles: int,
        out_h: int,
        out_w: int,
    ) -> np.ndarray:
        """Render each particle's view with barycentric vertex colors and SSIM-score it against the query."""
        # the kernel's interpolated uv is the hit's barycentric (u, v), see load_scene_geometry
        # uv_hits and hit_record are numba device arrays, as_tensor wraps them zero-copy
        uv = torch.as_tensor(uv_hits, dtype=torch.float32, device=self.device).reshape(-1, 2)
        hits = torch.as_tensor(hit_record, device=self.device).long()  # (N * rays,)
        tris = triangles[hits.clamp(min=0)].long()  # (N * rays, 3) vertex indices
        w = 1.0 - uv[:, 0:1] - uv[:, 1:2]
        rendered = (
            w * vertex_colors[tris[:, 0]]
            + uv[:, 0:1] * vertex_colors[tris[:, 1]]
            + uv[:, 1:2] * vertex_colors[tris[:, 2]]
        )
        rendered[hits < 0] = 0
        rendered = (
            rendered.reshape(n_particles, out_h, out_w, 3).permute(0, 3, 1, 2).float() / 255.0
        )
        return self.ssim_batch(rendered, query)

    @staticmethod
    def ssim_batch(x: torch.Tensor, y: torch.Tensor, win: int = 7) -> np.ndarray:
        """Compute the SSIM of each rendered view x (N,3,H,W) against the query y (1,3,H,W)."""
        n = x.shape[0]
        y = y.expand(n, -1, -1, -1)
        kernel = torch.ones(3, 1, win, win, dtype=x.dtype, device=x.device) / (win * win)

        def box(t: torch.Tensor) -> torch.Tensor:
            return F.conv2d(
                t, kernel, groups=3
            )  # per-channel win x win local mean (valid convolution)

        ux, uy = box(x), box(y)
        uxx, uyy, uxy = box(x * x), box(y * y), box(x * y)
        cov = (win * win) / (win * win - 1)  # sample covariance
        vx = cov * (uxx - ux * ux)
        vy = cov * (uyy - uy * uy)
        vxy = cov * (uxy - ux * uy)
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2)
        )
        return ssim_map.mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float32)

    def update_weights(self, query_image: int, pred_depth_map: np.ndarray) -> None:
        """Score every particle and write the normalized weights back onto the filter."""
        pf = self.pf
        likelihoods = self.score_particles(query_image, pred_depth_map, pf.poses)
        total = np.sum(likelihoods)
        if total > 0:
            pf.weights = likelihoods / total
        else:
            pf.weights = np.full_like(likelihoods, 1.0 / len(likelihoods))  # uniform fallback

    def score_particles(
        self, query_image: int, pred_depth_map: np.ndarray, poses: np.ndarray
    ) -> np.ndarray:
        """Combined semantic + depth + RGB likelihood for every particle pose."""
        pf = self.pf
        sim, predicted_node_ids = self.semantic_patch_labels(query_image)
        obj_ids = pf.scene.obj_ids
        raycaster = pf.raycaster
        batch_size = self.pose_batch_size(pf.num_pass)

        Rs = poses[:, :3, :3]  # cam-to-world rotations
        tvecs = poses[:, :3, 3:].squeeze(2)  # cam-to-world translations

        frame_idx = pf.data_dict["frame_idxs"][query_image]
        query_img_path = pf.query_image_path(pf.scan_id, frame_idx)
        stride = raycaster.stride
        ssim_h, ssim_w = self.img_height // stride, self.img_width // stride
        if raycaster.vertex_colors_gpu is not None:
            # vertex-colored mesh (ScanNet): barycentric interpolation, no texture
            vertex_colors_gpu = torch.as_tensor(
                raycaster.vertex_colors_gpu, device=self.device
            ).float()
            triangles_gpu = torch.as_tensor(raycaster.bvh_triangles_gpu, device=self.device)
        else:
            texture_img_path = pf.map_texture_path(pf.target_scan_id)
            tex_flat_gpu, tex_h, tex_w = self.load_texture_gpu(texture_img_path)
        query_gpu = self.load_query_gpu(query_img_path, ssim_h, ssim_w)
        # per-frame constants, computed and uploaded once for all particle batches
        obj_ids_gpu, node_ids_gpu, sim_gpu = raycaster.to_device_frame_inputs(
            obj_ids, predicted_node_ids, sim
        )
        pred_down, has_support = self.downsample_sensor_depth(pred_depth_map, (ssim_h, ssim_w))

        sem_scores, depth_scores, rgb_scores = [], [], []
        for start in range(0, len(Rs), batch_size):
            end = min(start + batch_size, len(Rs))
            n = end - start
            batch_sim, uv_hits, hit_record, depth_proj = raycaster.compute_rays_and_sim(
                Rs[start:end],
                tvecs[start:end],
                obj_ids_gpu,
                node_ids_gpu,
                sim_gpu,
            )
            sem_scores.append(batch_sim)
            depth_scores.append(self.depth_score(depth_proj, pred_down, has_support, n, stride))
            if raycaster.vertex_colors_gpu is not None:
                rgb = self.render_barycentric_and_ssim_gpu(
                    uv_hits,
                    hit_record,
                    vertex_colors_gpu,
                    triangles_gpu,
                    query_gpu,
                    n,
                    ssim_h,
                    ssim_w,
                )
            else:
                uv_hits = uv_hits.reshape(n, ssim_h * ssim_w, 2)
                rgb = self.render_and_ssim_gpu(
                    uv_hits, tex_flat_gpu, tex_h, tex_w, query_gpu, ssim_h, ssim_w
                )
            rgb_scores.append(self.score_max_per_term * rgb)  # SSIM [0, 1] -> [0, 2]

        combined = (
            np.concatenate(sem_scores)
            + np.concatenate(rgb_scores)
            + torch.cat(depth_scores).numpy()
        )
        return np.array(self.calc_likelihood(combined))

    def pose_batch_size(self, num_pass: int) -> int:
        obs = self.cfg.particle_filter.observation_model
        return [obs.first_pose_batch_size, obs.second_pose_batch_size, obs.third_pose_batch_size][
            num_pass - 1
        ]

    def semantic_patch_labels(self, query_image: int) -> tuple:
        """Return the per-patch top-1 object id (as flattened node ids) and its similarity."""
        pf = self.pf
        obj_ids = pf.obj_ids_slices[pf.scene_graph_to_index[pf.target_scan_id]]
        similarity_matrix = pf.patch_obj_sim_T[query_image] + 1  # shift to keep weights positive
        sim, obj_idxs = torch.topk(similarity_matrix, k=1, dim=1)
        sim = sim.numpy().flatten().astype(np.float32)
        predicted_node_ids = obj_ids[obj_idxs.numpy().flatten()]
        predicted_node_ids = np.flip(predicted_node_ids.reshape(self.patch_w, self.patch_h).T, 0)
        return sim, predicted_node_ids.flatten()

    @staticmethod
    def downsample_sensor_depth(pred_depth_map: np.ndarray, down_size: tuple) -> tuple:
        """Downsample the sensor depth ignoring holes and return (pred_down, has_support)."""
        pred = torch.as_tensor(pred_depth_map, dtype=torch.float32)
        valid = (pred > 1e-3).float()  # 0 = invalid sensor pixel
        weighted = (
            F.interpolate(
                (pred * valid).unsqueeze(0).unsqueeze(0),
                size=down_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        weight = (
            F.interpolate(
                valid.unsqueeze(0).unsqueeze(0),
                size=down_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
        pred_down = weighted / weight.clamp(min=1e-6)
        has_support = (weight > 0.1).float()  # downsampled pixels with sensor support
        return pred_down, has_support

    def depth_score(
        self,
        depth_proj: np.ndarray,
        pred_down: torch.Tensor,
        has_support: torch.Tensor,
        n_particles: int,
        stride: int,
    ) -> torch.Tensor:
        """Per-particle depth agreement between the raycast depth and the hole-aware sensor depth."""
        depth_proj = depth_proj.reshape(
            n_particles, self.img_height // stride, self.img_width // stride
        )
        # rays that miss the mesh carry the 1000 m penalty and saturate at the maximum penalty
        depth_proj = torch.tensor(depth_proj)
        diff = torch.clamp(
            pred_down - depth_proj, min=-self.depth_error_cap_m, max=self.depth_error_cap_m
        ).abs()
        diff = diff * has_support  # ignore sensor holes
        error = diff.sum(dim=(1, 2)) / has_support.sum().clamp(min=1.0)
        score = self.score_max_per_term - (
            self.score_max_per_term
            * torch.clamp(error, 0, self.depth_error_cap_m)
            / self.depth_error_cap_m
        )
        return torch.clamp(score, 0, self.score_max_per_term)
