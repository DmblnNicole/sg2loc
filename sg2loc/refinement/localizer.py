"""
Pose refinement: render views around the coarse pose, RoMa-match to the query, lift to 3D and
solve PnP. Writes frame_poses.txt (per frame: quaternion, translation, inlier count).
"""

from __future__ import annotations

import glob
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import poselib
import torch
import tqdm
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R

from sg2loc.particle_filter.raycasting import RayCasting
from sg2loc.particle_filter.scene import load_scene_geometry
from sg2loc.refinement.evaluation import evaluate_sequence
from sg2loc.utils import torch_util
from sg2loc.utils.torch_util import get_test_dataloader

logger = logging.getLogger(__name__)


def load_eval_scans(eval_scans_file) -> list:
    """Read the evaluation query scans, restrictable to a subset via the REFINEMENT_SCANS env var."""
    scans = [s.strip() for s in open(eval_scans_file) if s.strip()]
    subset_env = os.environ.get("REFINEMENT_SCANS")
    if subset_env:
        subset = [s.strip() for s in subset_env.split(",") if s.strip()]
        scans = [s for s in scans if s in subset] or subset
    return scans


def sequence_length_from_run(coarse_csv: str):
    """Read the sequence length from the sg2loc.yaml snapshot."""
    snapshot = os.path.join(os.path.dirname(coarse_csv), "sg2loc.yaml")
    if not os.path.exists(snapshot):
        return None
    with open(snapshot) as f:
        data = yaml.safe_load(f) or {}
    return data.get("particle_filter", {}).get("sequence_length")


def get_coarse_csv_path(cfg: Any, override: str = "") -> str:
    """Return the coarse-pose CSV to refine, the override when given, else the most recent particle-filter run."""
    if override:
        return override
    candidates = glob.glob(os.path.join(cfg.output_dir, "*", "filter", "sequence_poses_and_errors.csv"))
    if not candidates:
        raise OSError(
            f"no particle-filter run found under {cfg.output_dir}. Run the particle filter first or pass "
            "--coarse-csv with a sequence_poses_and_errors.csv."
        )
    return max(candidates, key=os.path.getmtime)


def load_coarse_pose_lookup(csv_path: str) -> dict:
    """Load the coarse-pose CSV into a dict keyed by (ScanID, int frame id)."""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["ScanID"]), int(row["FrameID"]))] = row
    return lookup


class RefinementLocalizer:
    """Drives pose refinement over the whole test set (one sequence per data_dict)."""

    def __init__(
        self,
        cfg: Any,
        roma_model: Any,
        pnp_thresh: int,
        coarse_csv: str,
        dataset_class: Any,
        refiner_class: Any,
        eval_scans_file,
        frame_mle_csv: str = "",
    ) -> None:
        self.cfg = cfg
        self.roma_model = roma_model
        self.pnp_thresh = pnp_thresh
        self.coarse_lookup = load_coarse_pose_lookup(coarse_csv)
        # optional second render anchor: the PF's per-frame MLE poses (dual-anchor refinement)
        self.frame_mle_lookup = load_coarse_pose_lookup(frame_mle_csv) if frame_mle_csv else None
        self.dataset_class = dataset_class
        self.refiner_class = refiner_class
        self.eval_scans = load_eval_scans(eval_scans_file)
        self.scene_cache: dict = {}
        self.appearance_cache: dict = {}

    def scene_for(self, target_scan_id: str):
        if target_scan_id not in self.scene_cache:
            bvh_dir = os.path.join(self.cfg.particle_filter.preprocess.output_dir, target_scan_id)
            self.scene_cache[target_scan_id] = load_scene_geometry(bvh_dir)
        return self.scene_cache[target_scan_id]

    def appearance_for(self, refiner_class: Any, target_scan_id: str):
        """Per-scene mesh appearance (texture image or per-triangle colors), cached."""
        if target_scan_id not in self.appearance_cache:
            self.appearance_cache.clear()  # sequences arrive grouped per scene, keep one
            self.appearance_cache[target_scan_id] = refiner_class.load_appearance(
                self.cfg, target_scan_id
            )
        return self.appearance_cache[target_scan_id]

    def run(self, out_dir: str) -> list:
        results_path = os.path.join(out_dir, "frame_poses.txt")
        open(results_path, "w").close()  # truncate stale rows from a prior run
        errors = []
        setup_t0 = time.perf_counter()
        with torch.no_grad():
            seq_idx = 0
            # scene selection is via eval_scans below (restrictable with REFINEMENT_SCANS)
            test_dataset, test_data_loader = get_test_dataloader(
                self.cfg, Dataset=self.dataset_class
            )
            self.test_data_loader = test_data_loader
            self.test_dataset = test_dataset
            data_dicts = tqdm.tqdm(
                enumerate(self.test_data_loader),
                total=len(self.test_data_loader),
                desc="refining query sequences",
            )
            self.setup_seconds = time.perf_counter() - setup_t0  # one-time dataset load
            loop_t0 = time.perf_counter()
            for _iteration, data_dict in data_dicts:  # one data_dict is one sequence
                # skip scan-boundary batches that mix frames from two scans
                is_target = (
                    len(set(data_dict["scan_ids"])) == 1
                    and data_dict["scan_ids"][0] in self.eval_scans
                )
                if not is_target:
                    continue
                self.scan_id = data_dict["scan_ids"][0]
                target_scan_id = data_dict["scan_ids_temp"][0]
                scene = self.scene_for(target_scan_id)
                appearance = self.appearance_for(self.refiner_class, target_scan_id)
                data_dict = torch_util.to_cuda(data_dict)
                refiner = self.refiner_class(
                    data_dict,
                    self.cfg,
                    self.coarse_lookup,
                    self.pnp_thresh,
                    scene,
                    appearance,
                    frame_mle_lookup=self.frame_mle_lookup,
                )
                refiner.out_dir = out_dir
                error = refiner.localize_seq(seq_idx, self.roma_model)
                if error is not None:
                    errors.append([len(errors), error[0], error[1]])
                seq_idx += 1
        self.loop_seconds = time.perf_counter() - loop_t0  # localization only (setup excluded)
        return errors


class SequenceRefiner:
    """Single-sequence refiner, subclassed per dataset to provide data access and rendering."""

    def __init__(
        self,
        data_dict: dict,
        cfg: Any,
        coarse_lookup: dict,
        pnp_thresh: int,
        scene: Any = None,
        appearance: Any = None,
        frame_mle_lookup: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_dict = data_dict
        self.coarse_lookup = coarse_lookup
        self.frame_mle_lookup = frame_mle_lookup
        self.pnp_thresh = pnp_thresh
        self.scene = scene
        self.appearance = appearance
        self.scan_id = data_dict["scan_ids"][0]  # scan id is always the same per data_dict
        self.target_scan_id = data_dict["scan_ids_temp"][0]  # rescan (we localize here)
        self.query_image = self.cfg.particle_filter.sequence_length - 1  # last frame in sequence
        self.scans_scenes_dir = self.cfg.particle_filter.scans_scenes_dir
        self.bvh_tree_dir = os.path.join(
            cfg.particle_filter.preprocess.output_dir, self.target_scan_id
        )
        self.img_width = cfg.data.img.w
        self.img_height = cfg.data.img.h
        # rendering uses the map scan's intrinsics, we localize in its frame
        self.intrinsics_rescan = self.load_intrinsics(self.target_scan_id)
        self.intrinsics_scan = self.load_intrinsics(self.scan_id)
        frame_idxs = self.data_dict["frame_idxs"]
        query_image_poses_scan = self.load_query_poses(self.scan_id, frame_idxs)
        self.query_image_poses_scan = query_image_poses_scan
        # align the query poses into the map scan frame
        recan_to_scan = self.map_to_query_transform(self.target_scan_id)
        scan_to_rescan = np.linalg.inv(recan_to_scan)
        query_image_poses_rescan = np.einsum("ij,njk->nik", scan_to_rescan, query_image_poses_scan)
        self.query_image_poses = query_image_poses_rescan

    @staticmethod
    def _scale_intrinsics(
        K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int
    ) -> np.ndarray:
        """Scale a pinhole K for a resize (non-uniform: fx/cx with x, fy/cy with y)."""
        sx, sy = dst_w / src_w, dst_h / src_h
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def calc_rel_pose(self, world_pose_cam1: np.ndarray, world_pose_cam2: np.ndarray) -> tuple:
        # relative pose of cam2 in the coordinate system of cam1, from cam-to-world poses
        cam2_pose_cam1 = np.linalg.inv(world_pose_cam1) @ world_pose_cam2
        cam2_rot_cam1 = cam2_pose_cam1[:3, :3]
        cam2_trans_cam1 = cam2_pose_cam1[:3, 3]
        return cam2_rot_cam1, cam2_trans_cam1

    def compute_2d_3d_correspondences_from_raycast(
        self,
        depth_map: np.ndarray,
        Rs: np.ndarray,
        tvecs: np.ndarray,
        img_width: int,
        img_height: int,
        rendered_kpts_dict: dict,
    ) -> list:
        """Lift the rendered keypoints to 3D and return (valid_indices, pts_3d) per pose."""
        num_poses = Rs.shape[0]
        correspondences_per_pose = []

        for pose_idx in range(num_poses):
            R = Rs[pose_idx]
            t = tvecs[pose_idx]
            kptsB = rendered_kpts_dict.get(f"db/rendered_{pose_idx + 1}.png", None)
            if kptsB is None or len(kptsB) == 0:
                correspondences_per_pose.append(
                    (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
                )
                continue

            valid_indices = []
            pts_3d = []

            for kp_idx, kp in enumerate(kptsB):
                j, i = kp[0], kp[1]  # x (col), y (row), pixel centers at k + 0.5
                j_pix = min(max(int(j), 0), img_width - 1)
                i_pix = min(max(int(i), 0), img_height - 1)
                idx = pose_idx * (img_width * img_height) + (i_pix * img_width + j_pix)
                z = depth_map[idx]
                if z >= 1000 or np.any(np.isnan(z)):
                    continue

                x = (j - self.intrinsics_rescan[0, 2]) / self.intrinsics_rescan[0, 0]
                y = (i - self.intrinsics_rescan[1, 2]) / self.intrinsics_rescan[1, 1]
                d_cam = np.array([x, y, 1.0])
                d_cam /= np.linalg.norm(d_cam)
                pt_cam = d_cam * z
                pt_world = R @ pt_cam + t

                valid_indices.append(kp_idx)
                pts_3d.append(pt_world)

            valid_indices = np.array(valid_indices, dtype=np.int32)
            pts_3d = np.array(pts_3d, dtype=np.float32)
            correspondences_per_pose.append((valid_indices, pts_3d))

        return correspondences_per_pose

    def _render_pose_hypotheses(self, mle_pose: np.ndarray) -> list:
        """Return the MLE render pose plus num_imgs views panned around the camera's gravity axis."""
        angles = np.linspace(
            self.cfg.refinement.min_angle,
            self.cfg.refinement.max_angle,
            self.cfg.refinement.num_imgs,
        )
        cam_axis = self.hypothesis_axis()
        render_poses = [mle_pose]
        camera_center = mle_pose[:3, 3]
        base_rotation = mle_pose[:3, :3].copy()
        gt_axis = base_rotation @ cam_axis
        gt_axis = gt_axis / np.linalg.norm(gt_axis)
        canonical = np.array([0, 0, -1])
        axis = np.cross(gt_axis, canonical)
        angle = np.arccos(np.clip(np.dot(gt_axis, canonical), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-8:
            align_to_canonical = np.eye(3)
        else:
            axis /= np.linalg.norm(axis)
            align_to_canonical = R.from_rotvec(angle * axis).as_matrix()
        aligned_base_rotation = align_to_canonical @ base_rotation
        for angle_deg in angles:
            roll = np.radians(angle_deg)
            R_roll = R.from_rotvec(roll * cam_axis).as_matrix()
            render_pose = np.eye(4)
            render_pose[:3, :3] = aligned_base_rotation @ R_roll
            render_pose[:3, 3] = camera_center
            render_poses.append(render_pose)
        return render_poses

    def _solve_pnp(self, mkpq: np.ndarray, mkp3d: np.ndarray) -> tuple:
        """Estimate the absolute pose via PnP RANSAC and return (qvec_xyzw, tvec, num_inliers)."""
        if len(mkpq) < 4:
            logger.warning(f"Too few correspondences ({len(mkpq)}), using identity + 0 inliers")
            return [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0
        # 2D points are from the scan camera, so use its intrinsics
        camera = {
            "model": "PINHOLE",
            "width": self.img_width,
            "height": self.img_height,
            "params": [
                self.intrinsics_scan[0][0],
                self.intrinsics_scan[1][1],
                self.intrinsics_scan[0][2],
                self.intrinsics_scan[1][2],
            ],
        }
        pose, info = poselib.estimate_absolute_pose(
            mkpq,
            mkp3d,
            camera,
            {"max_reproj_error": self.cfg.refinement.max_reproj_error},
            {},
        )
        if np.isnan(np.array(pose.q)).any():
            logger.warning("PnP failed, using identity + 0 inliers")
            return [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0
        q = pose.q  # wxyz
        return [q[1], q[2], q[3], q[0]], pose.t, info.get("num_inliers", -1)

    def localize_seq(self, seq_idx: int, roma_model: Any) -> tuple | None:
        # raycast, render, match and PnP at RoMa's internal resolution with scaled intrinsics
        roma_h, roma_w = int(roma_model.H_lr), int(roma_model.W_lr)
        self.intrinsics_rescan = self._scale_intrinsics(
            self.intrinsics_rescan, self.img_width, self.img_height, roma_w, roma_h
        )
        self.intrinsics_scan = self._scale_intrinsics(
            self.intrinsics_scan, self.img_width, self.img_height, roma_w, roma_h
        )
        self.img_width, self.img_height = roma_w, roma_h

        # BVH geometry and appearance come from the per-scene caches. Fall back for standalone use.
        if self.scene is None:
            self.scene = load_scene_geometry(self.bvh_tree_dir)
        if self.appearance is None:
            self.appearance = self.load_appearance(self.cfg, self.target_scan_id)
        raycaster = RayCasting(self.cfg, self.intrinsics_rescan, stride=1)
        raycaster.img_width, raycaster.img_height = self.img_width, self.img_height
        raycaster.set_scene(self.scene)

        # the coarse MLE pose belongs to the last frame of the sequence
        coarse_lookup = self.coarse_lookup
        last_frame_id = int(self.data_dict["frame_idxs"][-1])
        key = (str(self.scan_id), last_frame_id)
        if key not in coarse_lookup:
            logger.warning(f"no coarse MLE pose for {key} in the coarse CSV, skipping")
            return
        row = coarse_lookup[key]
        mle_pose_flat = row[[f"MLEPose_{i}" for i in range(16)]].values.astype(float)
        mle_pose = mle_pose_flat.reshape(4, 4)
        last_idx = self.query_image
        localized_raw = {}
        frame_inliers = {}
        for i in range(last_idx, -1, -1):
            # backpropagate the mle last-frame pose to frame i with the GT relative transform
            if i < last_idx:
                frame_key = (str(self.scan_id), int(self.data_dict["frame_idxs"][i]))
                if frame_key in coarse_lookup:
                    # per-frame coarse pose (frame_mle_poses.csv): use it directly, no propagation
                    row_i = coarse_lookup[frame_key]
                    mle_pose = (
                        row_i[[f"MLEPose_{j}" for j in range(16)]]
                        .values.astype(float)
                        .reshape(4, 4)
                    )
                else:
                    rel_rot, rel_trans = self.calc_rel_pose(
                        self.query_image_poses[i + 1], self.query_image_poses[i]
                    )
                    current_rot = mle_pose[:3, :3]
                    current_trans = mle_pose[:3, 3]
                    new_trans = current_trans + np.dot(current_rot, rel_trans)
                    new_rot = np.dot(current_rot, rel_rot)
                    mle_pose = np.eye(4)
                    mle_pose[:3, :3] = new_rot
                    mle_pose[:3, 3] = new_trans
            all_render_poses = self._render_pose_hypotheses(mle_pose)
            # dual-anchor: also render around the PF per-frame pose, one PnP over both anchors' views
            if self.frame_mle_lookup is not None:
                pf_row = self.frame_mle_lookup.get(
                    (str(self.scan_id), int(self.data_dict["frame_idxs"][i]))
                )
                if pf_row is not None:
                    pf_frame_pose = (
                        pf_row[[f"MLEPose_{j}" for j in range(16)]]
                        .values.astype(float)
                        .reshape(4, 4)
                    )
                    all_render_poses = all_render_poses + self._render_pose_hypotheses(
                        pf_frame_pose
                    )
            Rs = np.stack([pose[:3, :3] for pose in all_render_poses], axis=0)
            tvecs = np.stack([pose[:3, 3] for pose in all_render_poses], axis=0)

            uv_hits, hit_ids, depth_map = raycaster.cast_depth_uv(Rs, tvecs)

            # convert the raycast hits to RGB images (already at RoMa resolution, stride 1)
            rendered_imgs = []
            rays_per_pose = self.img_width * self.img_height
            for n in range(len(all_render_poses)):
                offset = n * rays_per_pose
                uv_hits_pose = uv_hits[offset : offset + rays_per_pose].reshape(
                    self.img_height, self.img_width, 2
                )
                hit_ids_pose = hit_ids[offset : offset + rays_per_pose]
                rendered_imgs.append(self.render_view(uv_hits_pose, hit_ids_pose))

            # load and resize the query to RoMa's resolution (the square RoMa matches at)
            query_img_idx = self.data_dict["frame_idxs"][i]
            query_img_path = self.query_image_path(self.scan_id, query_img_idx)
            query_rgb_np = np.asarray(Image.open(query_img_path)).astype(np.float32) / 255.0
            query_rgb_np = (
                cv2.resize(
                    (query_rgb_np * 255).astype(np.uint8),
                    (self.img_width, self.img_height),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)
                / 255.0
            )

            query_uint8 = (query_rgb_np * 255).astype(np.uint8)
            rendered_uint8 = [(img * 255).astype(np.uint8) for img in rendered_imgs]
            rendered_img_files = [f"rendered_{j + 1}.png" for j in range(len(rendered_imgs))]

            # debug: dump the exact RoMa inputs (query + each rendered view) per frame
            debug_dir = os.environ.get("REFINEMENT_DEBUG_RENDERS")
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                stem = f"{self.scan_id[:12]}_f{query_img_idx}"
                Image.fromarray(query_uint8).save(os.path.join(debug_dir, f"{stem}_query.png"))
                for j, u in enumerate(rendered_uint8):
                    Image.fromarray(u).save(os.path.join(debug_dir, f"{stem}_render{j}.png"))

            roma_matches_dict = {}
            rendered_kpts_dict = {}
            top_k = self.cfg.refinement.roma_topk

            # query and rendered views are all at RoMa resolution (img_height x img_width)
            H_A = H_B = self.img_height
            W_A = W_B = self.img_width
            # one batched forward per anchor group of rendered views
            query_t = torch.from_numpy(query_uint8).permute(2, 0, 1).unsqueeze(0).cuda()
            renders_t = torch.stack(
                [torch.from_numpy(u).permute(2, 0, 1) for u in rendered_uint8]
            ).cuda()
            chunk = self.cfg.refinement.num_imgs + 1
            preds_chunks = []
            for start in range(0, renders_t.shape[0], chunk):
                renders_c = renders_t[start : start + chunk]
                query_b = query_t.repeat(renders_c.shape[0], 1, 1, 1)
                preds_chunks.append(roma_model.match(query_b, renders_c))

            torch.manual_seed(0)

            for view_idx, db_img in enumerate(rendered_img_files):
                db_image_relpath = f"db/{db_img}"

                offset = view_idx % chunk
                preds = {
                    k: (v[offset : offset + 1] if torch.is_tensor(v) else v)
                    for k, v in preds_chunks[view_idx // chunk].items()
                }
                matches, overlaps, precision_AB, precision_BA = roma_model.sample(preds, top_k)

                if matches is None or len(matches) == 0:
                    continue

                kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                kptsA = kptsA.cpu().numpy()
                kptsB = kptsB.cpu().numpy()
                overlaps = overlaps.cpu().numpy()

                valid_mask = overlaps > self.cfg.refinement.match_certainty
                kptsA = kptsA[valid_mask]  # query
                kptsB = kptsB[valid_mask]  # db render
                overlaps = overlaps[valid_mask]

                rendered_kpts_dict[db_image_relpath] = kptsB
                roma_matches_dict[db_image_relpath] = {
                    "kptsA": kptsA,  # query
                    "kptsB": kptsB,  # db
                }

            del preds_chunks

            correspondences_db = {}

            correspondences_per_pose = self.compute_2d_3d_correspondences_from_raycast(
                depth_map,
                Rs,
                tvecs,
                self.img_width,
                self.img_height,
                rendered_kpts_dict,
            )
            for pose_idx, db_img in enumerate(rendered_img_files):
                valid_indices, points_3d = correspondences_per_pose[pose_idx]
                if len(valid_indices) == 0:
                    continue
                db_key = f"db/{db_img}"
                correspondences_db[db_key] = (valid_indices, points_3d)

            mkpq = []
            mkp3d = []

            for db_image in rendered_img_files:
                db_image_full = f"db/{db_image}"
                if (
                    db_image_full not in correspondences_db
                    or db_image_full not in roma_matches_dict
                ):
                    continue

                valid_indices, points_3d = correspondences_db[db_image_full]
                kptsA = roma_matches_dict[db_image_full]["kptsA"]

                # valid_indices aligns each 3D point with its matched query keypoint
                for j, idx in enumerate(valid_indices):
                    mkpq.append(kptsA[idx])
                    mkp3d.append(points_3d[j])

            mkpq = np.array(mkpq, dtype=np.float64).reshape(-1, 2)
            mkp3d = np.array(mkp3d, dtype=np.float64).reshape(-1, 3)

            results_path = os.path.join(self.out_dir, "frame_poses.txt")
            query_img_idx = self.data_dict["frame_idxs"][i]

            qvec, tvec, num_inliers = self._solve_pnp(mkpq, mkp3d)

            name = f"{self.scan_id} {query_img_idx}"
            with open(results_path, "a") as f:
                qvec_str = " ".join(map(str, qvec))
                tvec_str = " ".join(map(str, tvec))
                f.write(f"{name} {qvec_str} {tvec_str} {num_inliers}\n")

            frame_pose = np.eye(4)
            frame_pose[:3, :3] = R.from_quat(qvec).as_matrix()
            frame_pose[:3, 3] = tvec
            localized_raw[query_img_idx] = frame_pose
            frame_inliers[query_img_idx] = num_inliers

        frame_idxs = self.data_dict["frame_idxs"]
        gt_poses_rescan = dict(zip(frame_idxs, self.query_image_poses))
        return evaluate_sequence(
            frame_idxs, gt_poses_rescan, localized_raw, frame_inliers, row, self.pnp_thresh
        )
