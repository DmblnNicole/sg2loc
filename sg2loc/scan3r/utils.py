"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG
"""

import functools
import json
import os.path as osp
import pickle
from glob import glob

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_intrinsics(data_dir, scan_id, type="color"):
    """
    Load 3RScan intrinsic information
    """
    info_path = osp.join(data_dir, scan_id, "sequence", "_info.txt")

    width_search_string = "m_colorWidth" if type == "color" else "m_depthWidth"
    height_search_string = "m_colorHeight" if type == "color" else "m_depthHeight"
    calibration_search_string = (
        "m_calibrationColorIntrinsic" if type == "color" else "m_calibrationDepthIntrinsic"
    )

    with open(info_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.find(height_search_string) >= 0:
            intrinsic_height = line[line.find("= ") + 2 :]

        elif line.find(width_search_string) >= 0:
            intrinsic_width = line[line.find("= ") + 2 :]

        elif line.find(calibration_search_string) >= 0:
            intrinsic_mat = line[line.find("= ") + 2 :].split(" ")

            intrinsic_fx = intrinsic_mat[0]
            intrinsic_cx = intrinsic_mat[2]
            intrinsic_fy = intrinsic_mat[5]
            intrinsic_cy = intrinsic_mat[6]

            intrinsic_mat = np.array(
                [[intrinsic_fx, 0, intrinsic_cx], [0, intrinsic_fy, intrinsic_cy], [0, 0, 1]]
            )
            intrinsic_mat = intrinsic_mat.astype(np.float32)
    intrinsics = {
        "width": float(intrinsic_width),
        "height": float(intrinsic_height),
        "intrinsic_mat": intrinsic_mat,
    }

    return intrinsics


def load_pose(data_dir, scan_id, frame_id):
    pose_path = osp.join(data_dir, scan_id, "sequence", f"frame-{frame_id}.pose.txt")
    return np.genfromtxt(pose_path)


def load_all_poses(data_dir, scan_id, frame_idxs):
    frame_poses = []
    for frame_idx in frame_idxs:
        frame_pose = load_pose(data_dir, scan_id, frame_idx)
        frame_poses.append(frame_pose)
    frame_poses = np.array(frame_poses)
    return frame_poses


def load_frame_poses(data_dir, scan_id, frame_idxs, type="matrix"):
    frame_poses = {}
    for frame_idx in frame_idxs:
        frame_pose = load_pose(data_dir, scan_id, frame_idx)
        frame_pose = frame_pose.reshape(4, 4)
        if type == "matrix":
            frame_poses[frame_idx] = np.array(frame_pose)
        elif type == "quat_trans":
            T_pose = np.array(frame_pose)
            # transoformation matrix to quaternion+translation
            quaternion = R.from_matrix(T_pose[:3, :3]).as_quat()
            translation = T_pose[:3, 3]
            ## convert quaternion with translation to 7D vector
            frame_pose = np.concatenate([quaternion, translation])
        else:
            raise ValueError("Invalid type")

        frame_poses[frame_idx] = np.array(frame_pose)
    return frame_poses


def load_frame_idxs(data_dir, scan_id, skip=None):
    frames_paths = glob(osp.join(data_dir, scan_id, "sequence", "*.jpg"))
    frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
    frame_idxs = [frame_name.split(".")[0].split("-")[-1] for frame_name in frame_names]
    frame_idxs.sort()

    if skip is None:
        frame_idxs = frame_idxs
    else:
        frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]  # noqa: C416
    return frame_idxs


def load_frame_paths(data_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    img_folder = osp.join(data_dir, "scenes", scan_id, "sequence")
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = f"frame-{frame_idx}.color.jpg"
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths


def load_patch_feature_scans(data_root_dir, feature_folder, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_root_dir, "scenes"), scan_id, skip)
    features_file = osp.join(osp.join(data_root_dir, "files"), feature_folder, scan_id + ".pkl")
    with open(features_file, "rb") as handle:
        features_scan = pickle.load(handle)
    features_scan_step = {}
    for frame_idx in frame_idxs:
        features_scan_step[frame_idx] = features_scan[frame_idx]
    return features_scan_step


def sampleCandidateScenesForEachScan(scan_id, scan_ids, refscans2scans, scans2refscans, num_scenes):
    import random

    scans_same_scene = refscans2scans[scans2refscans[scan_id]]
    # sample other scenes
    sample_candidate_scans = [scan for scan in scan_ids if scan not in scans_same_scene]
    if num_scenes < 0:
        return sample_candidate_scans
    elif num_scenes <= len(sample_candidate_scans):
        return random.sample(sample_candidate_scans, num_scenes)
    else:
        return sample_candidate_scans


@functools.lru_cache(maxsize=1)
def load_rescan_transforms(json_path):
    """Rescan id -> reference scan + 4x4 rescan-to-scan transform, from 3RScan.json."""
    with open(json_path) as f:
        data = json.load(f)
    rescan_transforms = {}
    for entry in data:
        if "scans" not in entry:
            continue
        reference_scan = entry["reference"]
        for scan in entry["scans"]:
            rescan_id = scan["reference"]
            if "transform" in scan:
                # stored column-major, read row-major then transpose
                transform_matrix = np.array(scan["transform"]).reshape(4, 4).T
                rescan_transforms[rescan_id] = {
                    "reference_scan": reference_scan,
                    "transform_matrix": transform_matrix,
                }
    return rescan_transforms
