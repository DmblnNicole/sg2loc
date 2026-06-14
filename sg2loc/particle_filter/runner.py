"""
Shared (btw 3RScan and ScanNet) particle-filter runner that localizes every test sequence and prints the metrics.

The dataset entry points call run_main with their dataset class, particle filter class and
evaluation scene list. Writes sequence_poses_and_errors.csv, frame_data.csv and
frame_mle_poses.csv into the run directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import os
import os.path as osp
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy.spatial.transform import Rotation as R

from sg2loc.configs import config, snapshot_configs, update_config
from sg2loc.evaluation.pose_metrics import (
    frame_data_summary,
    print_pose_metrics,
    read_sequence_errors,
)
from sg2loc.models.patch_SGIE_aligner import PatchSGIEAligner
from sg2loc.particle_filter.scene import load_scene_geometry
from sg2loc.utils import common, torch_util
from sg2loc.utils.torch_util import get_test_dataloader

logger = logging.getLogger(__name__)


def _load_eval_scans(eval_scans_file) -> set[str]:
    """Read the evaluation query scans we localize from the dataset's repo scene list."""
    with open(eval_scans_file) as f:
        return {line.strip() for line in f if line.strip()}


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


class ParticleFilterRunner:
    """Drives particle-filter localization over the evaluation test set."""

    def __init__(
        self,
        cfg: Any,
        dataset_class: Any,
        filter_class: Any,
        eval_scans_file,
        only_scene: str = "",
        viz_dir: str = "",
        visualize_fn: Any = None,
    ) -> None:
        self.cfg = cfg
        self.method_name = cfg.val.room_retrieval.method_name
        self.filter_class = filter_class
        self.eval_scans = _load_eval_scans(eval_scans_file)
        self.only_scene = only_scene
        self.viz_dir = viz_dir
        self.visualize_fn = visualize_fn
        self.scene_cache: dict = {}  # target_scan_id -> SceneGeometry (BVH loaded/uploaded once)
        self.obj_embed_cache: dict = {}  # target_scan_id -> normalized 3D object embeddings

        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset=dataset_class)
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA devices available.")
        self.device = torch.device("cuda")

        self.build_model(cfg)
        self.model.eval()

        self.output_dir = osp.join(cfg.output_dir, self.method_name)
        common.ensure_dir(self.output_dir)

    def build_model(self, cfg: Any) -> None:
        """Build the SceneGraphLoc patch and object encoder."""
        backbone = None  # precomputed patch features are used (cfg.data.img_encoding.use_feature)

        multi_view_aggregator = getattr(cfg.sgaligner.model, "multi_view_aggregator", None)
        use_pos_enc = getattr(cfg.sgaligner, "use_pos_enc", False)
        self.use_temporal = cfg.train.loss.use_temporal
        self.use_global_descriptor = cfg.train.loss.use_global_descriptor
        self.global_descriptor_dim = cfg.model.global_descriptor_dim

        self.model = PatchSGIEAligner(
            backbone,
            cfg.model.backbone.num_reduce,
            cfg.model.backbone.backbone_dim,
            cfg.data.img_encoding.img_rotate,
            cfg.model.patch.hidden_dims,
            cfg.model.patch.encoder_dim,
            cfg.model.patch.gcn_layers,
            cfg.model.obj.embedding_dim,
            cfg.model.obj.embedding_hidden_dims,
            cfg.model.obj.encoder_dim,
            cfg.sgaligner.modules,
            cfg.sgaligner.model.rel_dim,
            cfg.sgaligner.model.attr_dim,
            cfg.sgaligner.model.img_patch_feat_dim,
            cfg.model.other.drop,
            self.use_temporal,
            self.use_global_descriptor,
            self.global_descriptor_dim,
            multi_view_aggregator=multi_view_aggregator,
            img_emb_dim=cfg.sgaligner.model.img_emb_dim,
            obj_img_pos_enc=use_pos_enc,
        )

        if cfg.sgaligner.use_pretrained:
            assert os.path.isfile(cfg.sgaligner.pretrained), "Pretrained sgaligner not found."
            sgaligner_dict = torch.load(cfg.sgaligner.pretrained, map_location=torch.device("cpu"))
            sgaligner_dict["model"].pop("fusion.weight")  # drop the last layer
            self.model.sg_encoder.load_state_dict(sgaligner_dict["model"], strict=False)

        if cfg.other.use_resume:
            assert os.path.isfile(cfg.other.resume), "Snapshot not found."
            self.load_snapshot(cfg.other.resume)

        self.model.to(self.device)
        self.model.eval()

    def load_snapshot(self, snapshot: str) -> None:
        state_dict = torch.load(snapshot, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["model"], strict=False)
        if "epoch" in state_dict:
            logger.info(f"model was trained for {state_dict['epoch']} epochs")
        else:
            logger.info("epoch information not available in the checkpoint")

    def encode_patches(self, data_dict: dict) -> torch.Tensor:
        """Encode the per-image patch features for the sequence, stacked to (B, P_H*P_W, C*)."""
        sequence_length = self.cfg.particle_filter.sequence_length
        if self.cfg.data.img_encoding.use_feature:
            features = data_dict["patch_features"][:sequence_length]
        else:
            images = _to_channel_first(data_dict["images"][:sequence_length])
            features = _to_channel_last(self.model.backbone(images)[-1])
        patch_features = self.model.reduce_layers(features)
        patch_features = self.model.patch_encoder(patch_features)
        patch_features = _to_channel_first(patch_features)
        patch_features = self.model.patch_gcn(patch_features)
        patch_features = _to_channel_last(patch_features)
        return patch_features.flatten(1, 2)

    def encode_scene_objects(self, data_dict: dict) -> torch.Tensor:
        """Return the L2-normalized 3D object embeddings, cached per target scene."""
        # the scene-graph input is fixed per target scene, so the encoder runs once per scene
        target_scan_id = data_dict["scan_ids_temp"][0]
        if target_scan_id not in self.obj_embed_cache:
            emb_path = osp.join(
                self.cfg.particle_filter.preprocess.embeddings_dir, f"{target_scan_id}.npy"
            )
            if osp.exists(emb_path):
                embeddings = torch.from_numpy(np.load(emb_path)).to(self.device)
            else:
                logger.warning(
                    f"no precomputed embeddings for {target_scan_id}, computing from raw features "
                    "(run the dataset's preprocessing/precompute_object_embeddings.py)"
                )
                obj_3d_embeddings = self.model.forward_scene_graph(data_dict)
                obj_3d_embeddings = self.model.obj_embedding_encoder(obj_3d_embeddings)
                embeddings = F.normalize(obj_3d_embeddings, dim=-1)
            self.obj_embed_cache[target_scan_id] = embeddings
        return self.obj_embed_cache[target_scan_id]

    @staticmethod
    def patch_obj_sims(assoc_data_dict: dict, obj_embeddings_norm, patch_features_norm):
        """Compute the cosine similarities between every patch and the target scan's objects."""
        candidates_obj_sg_idxs = assoc_data_dict["scans_sg_obj_idxs"]
        # the first candata_scan_obj_idxs entry is the target scan, distractor sims are never read
        target_scan_id = next(iter(assoc_data_dict["candata_scan_obj_idxs"]))
        target_obj_idxs = torch_util.release_cuda_torch(
            assoc_data_dict["candata_scan_obj_idxs"][target_scan_id]
        )
        obj_embeds_cpu = torch_util.release_cuda_torch(obj_embeddings_norm[candidates_obj_sg_idxs])
        return patch_features_norm @ obj_embeds_cpu[target_obj_idxs].T

    def get_patch_obj_sims(self, data_dict: dict) -> list:
        """Compute the patch to object similarities against the target scan for every sequence image."""
        patch_features_batch = self.encode_patches(data_dict)
        obj_embeddings_norm = self.encode_scene_objects(data_dict)
        sims = []
        for i in range(self.cfg.particle_filter.sequence_length):
            patch_features_norm = F.normalize(patch_features_batch[i].cpu(), dim=1)
            sims.append(
                self.patch_obj_sims(
                    data_dict["assoc_data_dict_temp"][i], obj_embeddings_norm, patch_features_norm
                )
            )
        return sims

    def run(self, out_dir: str) -> None:
        for data_dict in self.eval_sequences():
            self.localize_sequence(data_dict, out_dir)

    def eval_sequences(self):
        """Yield each test sequence on CUDA whose frames all come from one eval reference scan."""
        with torch.no_grad():
            loader = tqdm.tqdm(
                self.test_data_loader,
                total=len(self.test_data_loader),
                desc="localizing query sequences",
            )
            for data_dict in loader:  # one data_dict is one sequence
                scan_ids = data_dict["scan_ids"]
                # target_scan_id == query_scan_id == an eval reference scan, for every frame
                if len(set(scan_ids)) == 1 and scan_ids[0] in self.eval_scans:
                    if self.only_scene and not str(scan_ids[0]).startswith(self.only_scene):
                        continue
                    yield torch_util.to_cuda(data_dict)

    def localize_sequence(self, data_dict: dict, out_dir: str) -> None:
        # query scan we localize (same for all frames in the sequence)
        self.scan_id = data_dict["scan_ids"][0]
        target_scan_id = data_dict["scan_ids_temp"][0]
        if target_scan_id not in self.scene_cache:
            bvh_dir = osp.join(self.cfg.particle_filter.preprocess.output_dir, target_scan_id)
            self.scene_cache[target_scan_id] = load_scene_geometry(bvh_dir)
        sims = self.get_patch_obj_sims(data_dict)
        pf = self.filter_class(self.cfg, data_dict, sims, self.scene_cache[target_scan_id])
        top_pose, query_image_pose, per_frame_data = pf.run_particle_filter()

        if self.visualize_fn and pf.debug_snapshots:
            os.makedirs(self.viz_dir, exist_ok=True)
            seq = f"{self.scan_id[:8]}_{data_dict['frame_idxs'][-1]}"
            gif_path = osp.join(self.viz_dir, f"{seq}.gif")
            self.visualize_fn(self.cfg, self.scan_id, target_scan_id, pf.debug_snapshots, gif_path)
            logger.info(f"{seq}: {len(pf.debug_snapshots)} snapshots -> {gif_path}")

        position_error = np.linalg.norm(query_image_pose[:3, 3] - top_pose[:3, 3])
        rotation_diff = query_image_pose[:3, :3].T @ top_pose[:3, :3]
        rotation_error_deg = np.degrees(np.linalg.norm(R.from_matrix(rotation_diff).as_rotvec()))

        # frame id = the last query frame in the sequence (the one the MLE pose estimates)
        seq_frame_id = data_dict["frame_idxs"][-1]
        sequence_row = (rotation_error_deg, position_error, top_pose, seq_frame_id)
        self.write_csvs(sequence_row, per_frame_data, out_dir)

        # per-frame MLE poses from the final pass (one row per sequence frame)
        frame_rows = []
        for i, frame_id in enumerate(data_dict["frame_idxs"]):
            pose_i = pf.frame_mle_poses.get(frame_id)
            if pose_i is None:
                continue
            gt_i = pf.query_image_poses[i]
            pos_err = np.linalg.norm(gt_i[:3, 3] - pose_i[:3, 3])
            rot_diff = gt_i[:3, :3].T @ pose_i[:3, :3]
            rot_err = np.degrees(np.linalg.norm(R.from_matrix(rot_diff).as_rotvec()))
            frame_rows.append((frame_id, rot_err, pos_err, pose_i))
        self.write_frame_mle_csv(frame_rows, out_dir)

    def write_frame_mle_csv(self, frame_rows: list, out_dir: str) -> None:
        """Write one row per sequence frame with its final-pass MLE pose, schema matching the sequence CSV."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "frame_mle_poses.csv")
        write_header = not os.path.isfile(path)
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["ScanID", "FrameID", "RotationError", "PositionError"]
                    + [f"MLEPose_{i}" for i in range(16)]
                )
            for frame_id, rot_err, pos_err, pose in frame_rows:
                writer.writerow(
                    [self.scan_id, frame_id, f"{rot_err:.4f}", f"{pos_err:.4f}"]
                    + [f"{v:.4f}" for v in pose.flatten()]
                )

    def write_csvs(self, sequence_row: tuple, per_frame_data: list, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        frame_csv_path = os.path.join(out_dir, "frame_data.csv")
        sequence_csv_path = os.path.join(out_dir, "sequence_poses_and_errors.csv")

        write_header = not os.path.isfile(frame_csv_path)
        with open(frame_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["ScanID", "FrameID", "TimePerFrame", "ParticleNumber"])
            for frame_id, time_per_frame, particle_number in per_frame_data:
                writer.writerow([self.scan_id, frame_id, f"{time_per_frame:.4f}", particle_number])

        write_header = not os.path.isfile(sequence_csv_path)
        with open(sequence_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["ScanID", "FrameID", "RotationError", "PositionError"]
                    + [f"MLEPose_{i}" for i in range(16)]
                )
            rotation_error, position_error, top_pose, frame_id = sequence_row
            pose_flat = [f"{v:.4f}" for v in top_pose.flatten()]
            writer.writerow(
                [self.scan_id, frame_id, f"{rotation_error:.4f}", f"{position_error:.4f}"]
                + pose_flat
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str, help="configuration file name")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    parser.add_argument(
        "--sequence-length",
        default=0,
        type=int,
        help="frames per query sequence, minimum 2 "
        "(default: particle_filter.sequence_length from sg2loc.yaml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="render one debug GIF per sequence into <run>/viz",
    )
    parser.add_argument(
        "--debug-anchors",
        default="",
        type=str,
        help="with --debug: only render sequences with these anchor frames (comma-separated)",
    )
    parser.add_argument(
        "--scene",
        default="",
        type=str,
        help="only localize this scene, given as its scan id or a prefix of it",
    )
    args = parser.parse_args()
    if args.sequence_length and args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2 (the filter needs inter-frame motion)")
    return args


def run_main(
    dataset_class: Any, filter_class: Any, eval_scans_file, visualize_fn: Any = None
) -> None:
    """Entry point body, called by the dataset entry points with their classes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)  # silence CUDA dealloc chatter
    args = parse_args()
    cfg = update_config(config, args.config, ensure_dir=True, data_root=args.data_root)
    torch_util.set_seed(cfg.seed)
    if args.sequence_length:
        cfg.defrost()
        cfg.particle_filter.sequence_length = args.sequence_length
        cfg.freeze()
    logging.info(f"sequence length: {cfg.particle_filter.sequence_length} frames")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = osp.join(cfg.output_dir, timestamp, "filter")
    common.ensure_dir(out_dir)
    snapshot_configs(args.config, out_dir, cfg.particle_filter.sequence_length)
    if args.debug:
        os.environ["PF_DEBUG"] = "1"
        if args.debug_anchors:
            os.environ["PF_DEBUG_ANCHORS"] = args.debug_anchors

    runner = ParticleFilterRunner(
        cfg,
        dataset_class,
        filter_class,
        eval_scans_file,
        only_scene=args.scene,
        viz_dir=osp.join(cfg.output_dir, timestamp, "viz"),
        visualize_fn=visualize_fn,
    )
    runner.run(out_dir)

    errors = read_sequence_errors(osp.join(out_dir, "sequence_poses_and_errors.csv"))
    # per query frame: total localization time over all passes divided by the query frames
    total_time, avg_particles = frame_data_summary(osp.join(out_dir, "frame_data.csv"))
    n_query_frames = len(errors) * cfg.particle_filter.sequence_length
    time_per_frame = total_time / n_query_frames if total_time and n_query_frames else None
    print_pose_metrics(
        errors,
        "Particle filter (coarse poses)",
        time_per_frame,
        save_path=osp.join(out_dir, "metrics.txt"),
        avg_particles=avg_particles,
    )
