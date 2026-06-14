"""
Precomputes the per-scene object embeddings the particle filters semantic term uses.

The dataset entry points call precompute_main with their dataset class, particle filter
class and evaluation scene list. Writes one float32 .npy file per evaluation target scene
into preprocess.embeddings_dir, using the same encoder path as the runtime.
"""

import argparse
import logging
import os
from typing import Any

import numpy as np

from sg2loc.configs import config, update_config
from sg2loc.particle_filter.runner import ParticleFilterRunner
from sg2loc.utils.torch_util import set_seed

logger = logging.getLogger(__name__)


def precompute_main(dataset_class: Any, filter_class: Any, eval_scans_file) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("numba").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="backbone config (val.yaml)")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=True, data_root=args.data_root)
    set_seed(cfg.seed)
    out_dir = cfg.particle_filter.preprocess.embeddings_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = ParticleFilterRunner(cfg, dataset_class, filter_class, eval_scans_file)
    done = set()
    for data_dict in runner.eval_sequences():
        target_scan_id = data_dict["scan_ids_temp"][0]
        if target_scan_id in done:
            continue
        embeddings = runner.encode_scene_objects(data_dict)
        np.save(
            os.path.join(out_dir, f"{target_scan_id}.npy"),
            embeddings.detach().cpu().numpy().astype(np.float32),
        )
        done.add(target_scan_id)
        logger.info(f"{target_scan_id}: {tuple(embeddings.shape)}")
    logger.info(f"saved embeddings for {len(done)} scene(s) to {out_dir}")
