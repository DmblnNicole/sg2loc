"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.

Usage:
    python -m sg2loc.scannet.preprocessing.convert_sgfusion_to_scan3r \
        --scans-dir /path/to/scannet/scans \
        --scene-list sg2loc/scannet/preprocessing/scene_lists/scannet_eval_query_scans.txt
"""

import argparse
import os.path as osp

from yacs.config import CfgNode as CN

from sg2loc.scannet import utils as scannet
from sg2loc.utils import common

LABELS_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans-dir", required=True, help="directory with the ScanNet scans")
    parser.add_argument("--scene-list", required=True, help="txt file, one scan id per line")
    parser.add_argument("--filter-segment-size", type=int, default=512)
    parser.add_argument("--min-obj-points", type=int, default=50)
    args = parser.parse_args()

    rel2idx = common.name2idx(osp.join(LABELS_DIR, "scannet8_relationships.txt"))
    class2idx = common.name2idx(osp.join(LABELS_DIR, "scannet20_classes.txt"))

    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.preprocess = CN()
    cfg.preprocess.set_new_allowed(True)
    cfg.preprocess.filter_segment_size = args.filter_segment_size
    cfg.preprocess.min_obj_points = args.min_obj_points
    cfg.preprocess.pc_resolutions = [64, 128, 256, 512]

    scan_ids = [ln.strip() for ln in open(args.scene_list) if ln.strip()]
    # the list holds the _00 query scans, each room's _01 map twin is processed too
    scan_ids = [s for q in scan_ids for s in (q, q[:-3] + "_01")]
    converted, skipped = 0, []
    for scan_id in scan_ids:
        pred_folder = osp.join(args.scans_dir, scan_id, "scene_graph_fusion")
        if not osp.isfile(osp.join(pred_folder, "predictions.json")):
            skipped.append(scan_id)
            continue
        data_dict = scannet.scenegraphfusion2scan3r(scan_id, pred_folder, rel2idx, class2idx, cfg)
        # scenegraphfusion2scan3r returns -1 when the scene filters down to <2 objects
        if isinstance(data_dict, int):
            skipped.append(scan_id)
            print(f"[sg2scan3r] skip {scan_id}: degenerate scene graph")
            continue
        pkl = osp.join(pred_folder, f"{scan_id}.pkl")
        common.write_pkl_data(data_dict, pkl)
        scannet.calculate_bow_node_edge_feats(pkl, rel2idx)
        converted += 1
    print(
        f"[sg2scan3r] converted {converted}/{len(scan_ids)}, skipped: {skipped if skipped else 'none'}"
    )


if __name__ == "__main__":
    main()
