"""
Shared (btw ScanNet and 3RScan) pose-refinement runner that localizes each sequence, evaluates it and prints the metrics.

The dataset entry points call run_main with their dataset class, sequence refiner class and
evaluation scene list. --coarse-csv selects the particle-filter poses to refine, defaulting
to the most recent particle-filter run.
"""

import argparse
import datetime
import logging
import os
from typing import Any

import torch
from romav2 import RoMaV2

from sg2loc.configs import config, snapshot_configs, update_config
from sg2loc.evaluation.pose_metrics import print_pose_metrics
from sg2loc.refinement.evaluation import write_errors_csv
from sg2loc.refinement.localizer import (
    RefinementLocalizer,
    get_coarse_csv_path,
    sequence_length_from_run,
)
from sg2loc.utils.torch_util import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", default="", type=str, help="configuration file name"
    )
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    parser.add_argument(
        "--coarse-csv",
        default="",
        type=str,
        help="particle-filter sequence_poses_and_errors.csv to refine (default: the most recent run)",
    )
    return parser.parse_args()


def run_main(dataset_class: Any, refiner_class: Any, eval_scans_file) -> None:
    """Entry point body, called by the dataset entry points with their classes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)  # silence CUDA dealloc chatter
    logging.getLogger("romav2").setLevel(logging.WARNING)  # its rich markup prints literally here
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    args = parse_args()
    cfg = update_config(config, args.config, ensure_dir=True, data_root=args.data_root)
    set_seed(cfg.seed)

    # the refinement run dir lives next to the pf dir of the coarse run it refines
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    coarse_csv = get_coarse_csv_path(cfg, args.coarse_csv)
    logger.info(f"coarse poses: {coarse_csv}")
    # the sequence length is a property of the coarse run, read it from its config snapshot
    seq_len = sequence_length_from_run(coarse_csv)
    if not seq_len:
        raise OSError(
            "no sg2loc.yaml snapshot next to the coarse CSV. The sequence length must come "
            "from the coarse run itself, a mismatch would garble the sequence grouping."
        )
    cfg.defrost()
    cfg.particle_filter.sequence_length = seq_len
    cfg.freeze()
    logger.info(f"sequence length: {seq_len} frames (from the coarse run)")
    coarse_dir = os.path.dirname(coarse_csv)
    if os.path.basename(coarse_dir) == "filter":
        run_dir = os.path.join(os.path.dirname(coarse_dir), f"refine_{timestamp}")
    else:
        run_dir = os.path.join(cfg.output_dir, f"refine_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    snapshot_configs(args.config, run_dir, cfg.particle_filter.sequence_length)

    results_path = os.path.join(run_dir, "frame_poses.txt")
    pnp_thresh = int(os.environ.get("REFINEMENT_THRESH", cfg.refinement.pnp_thresh))

    logger.info("Loading RoMaV2 model...")
    roma_model = RoMaV2(RoMaV2.Cfg(setting=cfg.refinement.roma_setting))
    logger.info(
        f"RoMaV2 model loaded ({cfg.refinement.roma_setting}: "
        f"{roma_model.W_lr}x{roma_model.H_lr} coarse, H_hr={roma_model.H_hr})"
    )

    frame_mle_csv = os.path.join(os.path.dirname(coarse_csv), "frame_mle_poses.csv")
    frame_mle_csv = frame_mle_csv if os.path.isfile(frame_mle_csv) else ""

    # localize each sequence and evaluate it inline (writes frame_poses.txt)
    localizer = RefinementLocalizer(
        cfg,
        roma_model,
        pnp_thresh,
        coarse_csv,
        dataset_class,
        refiner_class,
        eval_scans_file,
        frame_mle_csv=frame_mle_csv,
    )
    errors = localizer.run(run_dir)

    write_errors_csv(run_dir, errors)

    n_frames = sum(1 for _ in open(results_path))
    time_per_frame = localizer.loop_seconds / n_frames if n_frames else None
    logger.info(f"one-time setup (dataset load): {localizer.setup_seconds:.1f} s")
    print_pose_metrics(
        [(e[1], e[2]) for e in errors],
        "pose refinement",
        time_per_frame,
        save_path=os.path.join(run_dir, "metrics.txt"),
    )
