"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG (src/datasets/scannet_objpair.py)
"""

import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data

from sg2loc.scannet import utils as scannet
from sg2loc.utils import common, point_cloud

torch.multiprocessing.set_sharing_strategy("file_system")

# the 48 evaluation query scans (the _00 scans that yield data items)
EVAL_SCANS_FILE = (
    Path(__file__).resolve().parent
    / "preprocessing"
    / "scene_lists"
    / "scannet_eval_query_scans.txt"
)


class ScannetPatchObjDataset(data.Dataset):
    """ScanNet counterpart of PatchObjectPairXTAESGIDataSet (eval splits only)."""

    def __init__(self, cfg, split, num_test_data_set):
        self.cfg = cfg
        self.undefined = 0  # patch anno id for unlabeled patches

        self.split = split
        self.num_test_data_set = num_test_data_set
        self.sgaligner_modules = cfg.sgaligner.modules

        # rooms info: room id -> [scan ids] (48 test rooms, one _00 and one _01 scan each)
        self.data_root_dir = cfg.data.root_dir
        self.scans_files_dir = osp.join(self.data_root_dir, "files")
        rooms_info = common.load_pkl_data(osp.join(self.scans_files_dir, f"scans_{split}.pkl"))
        self.scan2room = {}
        self.room2scans = rooms_info
        self.scan_ids = []
        for room_id in rooms_info:
            self.scan_ids += rooms_info[room_id]
            for scan_id in rooms_info[room_id]:
                self.scan2room[scan_id] = room_id

        # evaluation query scans (the _00 scans that yield data items)
        self.test_ref_scans = {line.strip() for line in open(EVAL_SCANS_FILE) if line.strip()}

        # cross scenes cfg
        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes

        # 2D frames and patch features (query scans only)
        self.scans_scenes_dir = osp.join(self.data_root_dir, cfg.particle_filter.scans_scenes_dir)
        self.img_step = cfg.data.img.img_step
        self.patch_h = cfg.data.img_encoding.patch_h
        self.patch_w = cfg.data.img_encoding.patch_w
        self.num_patch = self.patch_h * self.patch_w
        self.img_patch_feat_dim = cfg.sgaligner.model.img_patch_feat_dim
        feature_folder = osp.join(self.scans_files_dir, cfg.data.img_encoding.feature_dir)
        self.frame_idxs = {}
        self.features_path = {}
        for scan_id in self.test_ref_scans:
            frame_idxs = scannet.load_frame_idxs(self.scans_scenes_dir, scan_id, self.img_step)
            self.frame_idxs[scan_id] = frame_idxs
            self.features_path[scan_id] = {
                frame_idx: osp.join(feature_folder, scan_id, f"{frame_idx}.npy")
                for frame_idx in frame_idxs
            }

        # patch anno, keeping only frames with enough labeled patches and a valid GT pose
        patch_anno_folder = osp.join(self.scans_files_dir, cfg.data.gt_patch)
        patch_anno_th = cfg.data.gt_patch_th
        self.patch_anno = {}
        for scan_id in self.test_ref_scans:
            patch_anno_scan = common.load_pkl_data(osp.join(patch_anno_folder, f"{scan_id}.pkl"))
            poses = scannet.load_frame_poses(self.scans_scenes_dir, scan_id)
            self.patch_anno[scan_id] = {}
            for frame_idx in self.frame_idxs[scan_id]:
                if frame_idx not in patch_anno_scan or frame_idx not in poses:
                    continue
                num_valid = np.sum(patch_anno_scan[frame_idx] != self.undefined)
                if num_valid / self.num_patch > patch_anno_th:
                    self.patch_anno[scan_id][frame_idx] = patch_anno_scan[frame_idx]

        # 3D scene graphs (SceneGraphFusion predictions converted to the scan3r format)
        self.load3DSceneGraphs()

        # per-object visual embeddings for the img_patch module
        obj_img_patch_name = cfg.data.scene_graph.obj_img_patch
        self.obj_topk = cfg.data.scene_graph.obj_topk
        self.obj_img_patches_scan_tops = {}
        if "img_patch" in self.sgaligner_modules:
            for scan_id in self.scan_ids:
                obj_visual_file = osp.join(
                    self.scans_files_dir, obj_img_patch_name, f"{scan_id}.pkl"
                )
                self.obj_img_patches_scan_tops[scan_id] = common.load_pkl_data(obj_visual_file)

        # fixed candidate scans per target scan (empty with cross_scene.num_scenes = 0)
        self.candidate_scans = {}
        for scan_id in self.scan_ids:
            candidates = [s for s in self.scan_ids if self.scan2room[s] != self.scan2room[scan_id]]
            self.candidate_scans[scan_id] = random.sample(candidates, self.num_scenes)

        self.data_items = self.generateDataItems()

    def load3DSceneGraphs(self):
        self.pc_resolution = self.cfg.sgaligner.val.pc_res
        rel_dim = self.cfg.sgaligner.model.rel_dim
        self.scene_graphs = {}
        self.obj_3D_anno = {}
        for scan_id in self.scan_ids:
            sg_folder_scan = osp.join(self.scans_scenes_dir, scan_id, "scene_graph_fusion")
            # centering
            points = point_cloud.load_plydata_npy(
                osp.join(sg_folder_scan, "data.npy"), obj_ids=None
            )
            pcl_center = np.mean(points, axis=0)
            # scene graph info
            scene_graph_dict = common.load_pkl_data(osp.join(sg_folder_scan, f"{scan_id}.pkl"))
            object_ids = scene_graph_dict["objects_id"]
            global_object_ids = scene_graph_dict["objects_cat"]
            object_points = scene_graph_dict["obj_points"][self.pc_resolution] - pcl_center
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
            edges = torch.from_numpy(scene_graph_dict["edges"])
            bow_vec_obj_edge_feats = torch.from_numpy(scene_graph_dict["bow_vec_object_edge_feats"])
            rel_pose = torch.from_numpy(scene_graph_dict["rel_trans"])

            data_dict = {}
            data_dict["obj_ids"] = object_ids
            data_dict["tot_obj_pts"] = object_points
            data_dict["graph_per_obj_count"] = np.array([object_points.shape[0]])
            data_dict["graph_per_edge_count"] = np.array([edges.shape[0]])
            data_dict["tot_obj_count"] = object_points.shape[0]
            data_dict["tot_bow_vec_object_edge_feats"] = bow_vec_obj_edge_feats
            # SceneGraphFusion predictions carry no attribute annotations
            data_dict["tot_bow_vec_object_attr_feats"] = torch.zeros(
                object_points.shape[0], rel_dim
            )
            data_dict["tot_rel_pose"] = rel_pose
            data_dict["edges"] = edges
            data_dict["global_obj_ids"] = global_object_ids
            data_dict["scene_ids"] = [scan_id]
            data_dict["pcl_center"] = pcl_center
            self.scene_graphs[scan_id] = data_dict

            self.obj_3D_anno[scan_id] = {}
            for idx, obj_id in enumerate(object_ids):
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, global_object_ids[idx])

    def targetScanOfRoom(self, scan_id):
        """The map scan the query localizes in (the other scan of the same room)."""
        others = [s for s in self.room2scans[self.scan2room[scan_id]] if s != scan_id]
        return others[self.num_test_data_set]

    def generateDataItems(self):
        data_items = []
        for scan_id in sorted(self.test_ref_scans):
            target_scan_id = self.targetScanOfRoom(scan_id)
            for frame_idx in self.frame_idxs[scan_id]:
                if frame_idx not in self.patch_anno[scan_id]:
                    continue
                data_items.append(
                    {
                        "scan_id": scan_id,
                        "scan_id_temporal": target_scan_id,
                        "frame_idx": frame_idx,
                        "patch_feature_path": self.features_path[scan_id][frame_idx],
                    }
                )
        return data_items

    def dataItem2DataDict(self, data_item):
        scan_id = data_item["scan_id"]
        frame_idx = data_item["frame_idx"]
        patch_anno_frame = self.patch_anno[scan_id][frame_idx]
        obj_2D_patch_anno_flatten = patch_anno_frame.reshape(-1).astype(np.int32)
        patch_feature = np.load(data_item["patch_feature_path"])
        if patch_feature.ndim == 2:
            patch_feature = patch_feature.reshape(
                self.patch_h, self.patch_w, self.img_patch_feat_dim
            )
        return {
            "scan_id": scan_id,
            "scan_id_temporal": data_item["scan_id_temporal"],
            "frame_idx": frame_idx,
            "patch_features": patch_feature,
            "obj_2D_patch_anno_flatten": obj_2D_patch_anno_flatten,
        }

    def generateObjPatchAssociationScan(
        self, scan_id, candidate_scans, gt_2D_anno_flat, sg_obj_idxs
    ):
        obj_3D_idx2info = {}
        obj_3D_id2idx_cur_scan = {}
        scans_sg_obj_idxs = []
        candata_scan_obj_idxs = {}

        ## cur scan objs
        idx = 0
        for obj_id in self.scene_graphs[scan_id]["obj_ids"]:
            obj_3D_idx2info[idx] = self.obj_3D_anno[scan_id][obj_id]
            obj_3D_id2idx_cur_scan[obj_id] = idx
            scans_sg_obj_idxs.append(sg_obj_idxs[scan_id][obj_id])
            candata_scan_obj_idxs.setdefault(scan_id, []).append(idx)
            idx += 1
        ## other scans objs
        for cand_scan_id in candidate_scans:
            for obj_id in self.scene_graphs[cand_scan_id]["obj_ids"]:
                obj_3D_idx2info[idx] = self.obj_3D_anno[cand_scan_id][obj_id]
                scans_sg_obj_idxs.append(sg_obj_idxs[cand_scan_id][obj_id])
                candata_scan_obj_idxs.setdefault(cand_scan_id, []).append(idx)
                idx += 1
        for cand_scan_id in candata_scan_obj_idxs:
            candata_scan_obj_idxs[cand_scan_id] = torch.Tensor(
                candata_scan_obj_idxs[cand_scan_id]
            ).long()
        scans_sg_obj_idxs = torch.from_numpy(np.array(scans_sg_obj_idxs, dtype=np.int32)).long()

        ## generate obj patch association
        ## e1i_matrix (num_patch, num_3D_obj): 2D-3D patch-object pairs
        ## e2j_matrix (num_patch, num_3D_obj): 2D-3D patch-object unpairs
        num_objs = idx
        e1i_matrix = np.zeros((self.num_patch, num_objs), dtype=np.uint8)
        e2j_matrix = np.ones((self.num_patch, num_objs), dtype=np.uint8)
        for patch_idx in range(self.num_patch):
            obj_id = gt_2D_anno_flat[patch_idx]
            if obj_id != self.undefined and (obj_id in obj_3D_id2idx_cur_scan):
                obj_idx = obj_3D_id2idx_cur_scan[obj_id]
                e1i_matrix[patch_idx, obj_idx] = 1
                e2j_matrix[patch_idx, obj_idx] = 0
        ## e1j_matrix (num_patch, num_patch): unpaired patch-patch pairs
        e1j_matrix = np.zeros((self.num_patch, self.num_patch), dtype=np.uint8)
        for patch_idx in range(self.num_patch):
            obj_id = gt_2D_anno_flat[patch_idx]
            if obj_id != self.undefined and obj_id in obj_3D_id2idx_cur_scan:
                e1j_matrix[patch_idx, :] = np.logical_and(
                    gt_2D_anno_flat != self.undefined, gt_2D_anno_flat != obj_id
                )
            else:
                e1j_matrix[patch_idx, :] = 1
        ## f1j_matrix (num_3D_obj, num_3D_obj): objects of different semantic category
        obj_cates_arr = np.array(
            [obj_3D_idx2info[obj_idx][2] for obj_idx in range(len(obj_3D_idx2info))]
        )
        f1j_matrix = obj_cates_arr.reshape(1, -1) != obj_cates_arr.reshape(-1, 1)

        return {
            "e1i_matrix": torch.from_numpy(e1i_matrix).float(),
            "e1j_matrix": torch.from_numpy(e1j_matrix).float(),
            "e2j_matrix": torch.from_numpy(e2j_matrix).float(),
            "f1j_matrix": torch.from_numpy(f1j_matrix).float(),
            "scans_sg_obj_idxs": scans_sg_obj_idxs,
            "candata_scan_obj_idxs": candata_scan_obj_idxs,
        }

    def aggretateDataDicts(self, data_dict, key, mode):
        if mode == "torch_cat":
            return torch.cat([data[key] for data in data_dict])
        elif mode == "np_concat":
            return np.concatenate([data[key] for data in data_dict])
        elif mode == "np_stack":
            return np.stack([data[key] for data in data_dict])
        else:
            raise NotImplementedError

    def collateBatchDicts(self, batch):
        # the scene graphs the model encodes: the target (map) scans plus their candidates
        scans_batch = [data["scan_id_temporal"] for data in batch]
        if self.use_cross_scene:
            candidate_scans = {scan_id: self.candidate_scans[scan_id] for scan_id in scans_batch}
            union_scans = list(
                set(
                    scans_batch
                    + [scan for scan_list in candidate_scans.values() for scan in scan_list]
                )
            )
        else:
            candidate_scans, union_scans = None, list(set(scans_batch))

        data_dict = {}
        data_dict["batch_size"] = len(batch)
        data_dict["temporal"] = True
        data_dict["scan_ids"] = np.stack([data["scan_id"] for data in batch])
        data_dict["scan_ids_temp"] = np.stack([data["scan_id_temporal"] for data in batch])
        data_dict["frame_idxs"] = np.stack([data["frame_idx"] for data in batch])
        patch_features_batch = np.stack([data["patch_features"] for data in batch])
        data_dict["patch_features"] = torch.from_numpy(patch_features_batch).float()
        data_dict["obj_2D_patch_anno_flatten_list"] = [
            torch.from_numpy(data["obj_2D_patch_anno_flatten"]) for data in batch
        ]

        # 3D scene graph info
        scene_graph_infos = [self.scene_graphs[scan_id] for scan_id in union_scans]
        scene_graphs_ = {}
        scene_graphs_["batch_size"] = len(scene_graph_infos)
        scene_graphs_["obj_ids"] = self.aggretateDataDicts(
            scene_graph_infos, "obj_ids", "np_concat"
        )
        scene_graphs_["tot_obj_pts"] = self.aggretateDataDicts(
            scene_graph_infos, "tot_obj_pts", "torch_cat"
        )
        scene_graphs_["graph_per_obj_count"] = self.aggretateDataDicts(
            scene_graph_infos, "graph_per_obj_count", "np_stack"
        )
        scene_graphs_["graph_per_edge_count"] = self.aggretateDataDicts(
            scene_graph_infos, "graph_per_edge_count", "np_stack"
        )
        scene_graphs_["tot_obj_count"] = self.aggretateDataDicts(
            scene_graph_infos, "tot_obj_count", "np_stack"
        )
        scene_graphs_["tot_bow_vec_object_attr_feats"] = self.aggretateDataDicts(
            scene_graph_infos, "tot_bow_vec_object_attr_feats", "torch_cat"
        ).double()
        scene_graphs_["tot_bow_vec_object_edge_feats"] = self.aggretateDataDicts(
            scene_graph_infos, "tot_bow_vec_object_edge_feats", "torch_cat"
        ).double()
        scene_graphs_["tot_rel_pose"] = self.aggretateDataDicts(
            scene_graph_infos, "tot_rel_pose", "torch_cat"
        ).double()
        scene_graphs_["edges"] = self.aggretateDataDicts(scene_graph_infos, "edges", "torch_cat")
        scene_graphs_["global_obj_ids"] = self.aggretateDataDicts(
            scene_graph_infos, "global_obj_ids", "np_concat"
        )
        scene_graphs_["scene_ids"] = self.aggretateDataDicts(
            scene_graph_infos, "scene_ids", "np_stack"
        )
        scene_graphs_["pcl_center"] = self.aggretateDataDicts(
            scene_graph_infos, "pcl_center", "np_stack"
        )
        ### per-object image patch embeddings
        if "img_patch" in self.sgaligner_modules:
            obj_img_patches = {}
            obj_count_ = 0
            for scan_idx, scan_id in enumerate(scene_graphs_["scene_ids"]):
                scan_id = scan_id[0]
                obj_start_idx = obj_count_
                obj_end_idx = obj_count_ + scene_graphs_["tot_obj_count"][scan_idx]
                obj_ids = scene_graphs_["obj_ids"][obj_start_idx:obj_end_idx]
                obj_img_patches_scan_tops = self.obj_img_patches_scan_tops[scan_id]
                obj_img_patches_scan = obj_img_patches_scan_tops["obj_visual_emb"]
                obj_top_frames = obj_img_patches_scan_tops["obj_image_votes_topK"]

                obj_img_patches[scan_id] = {}
                for obj_id in obj_ids:
                    if obj_id not in obj_top_frames:
                        obj_img_patches[scan_id][obj_id] = torch.zeros(
                            1, self.img_patch_feat_dim
                        ).float()
                        continue
                    obj_img_patch_embs_list = []
                    obj_frames = obj_top_frames[obj_id][: self.obj_topk]
                    for frame_idx in obj_frames:
                        if obj_img_patches_scan[obj_id][frame_idx] is not None:
                            embs_frame = obj_img_patches_scan[obj_id][frame_idx]
                            embs_frame = (
                                embs_frame.reshape(1, -1) if embs_frame.ndim == 1 else embs_frame
                            )
                            obj_img_patch_embs_list.append(embs_frame)
                    if len(obj_img_patch_embs_list) == 0:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                    else:
                        obj_img_patch_embs = np.concatenate(obj_img_patch_embs_list, axis=0)
                    obj_img_patches[scan_id][obj_id] = torch.from_numpy(obj_img_patch_embs).float()
                obj_count_ += scene_graphs_["tot_obj_count"][scan_idx]
            scene_graphs_["obj_img_patches"] = obj_img_patches

        data_dict["scene_graphs"] = scene_graphs_

        ## per-frame association against the target scan graph
        sg_obj_idxs = {}
        sg_obj_idxs_tensor = {}
        sg_obj_idx_start = 0
        for scan_idx, scan_id in enumerate(scene_graphs_["scene_ids"]):
            scan_id = scan_id[0]
            sg_obj_idxs[scan_id] = {}
            objs_count = scene_graphs_["tot_obj_count"][scan_idx]
            sg_obj_idxs_tensor[scan_id] = torch.from_numpy(
                np.arange(sg_obj_idx_start, sg_obj_idx_start + objs_count)
            ).long()
            for sg_obj_idx in range(sg_obj_idx_start, sg_obj_idx_start + objs_count):
                obj_id = scene_graphs_["obj_ids"][sg_obj_idx]
                sg_obj_idxs[scan_id][obj_id] = sg_obj_idx
            sg_obj_idx_start += objs_count
        assoc_data_dict_temporal = []
        for item in batch:
            target_scan_id = item["scan_id_temporal"]
            candidate_scans_cur = [] if candidate_scans is None else candidate_scans[target_scan_id]
            assoc_data_dict_temporal.append(
                self.generateObjPatchAssociationScan(
                    target_scan_id,
                    candidate_scans_cur,
                    item["obj_2D_patch_anno_flatten"],
                    sg_obj_idxs,
                )
            )
        data_dict["assoc_data_dict_temp"] = assoc_data_dict_temporal
        data_dict["sg_obj_idxs"] = sg_obj_idxs
        data_dict["sg_obj_idxs_tensor"] = sg_obj_idxs_tensor
        data_dict["candidate_scans"] = candidate_scans
        return data_dict if len(batch) > 0 else None

    def __getitem__(self, idx):
        return self.dataItem2DataDict(self.data_items[idx])

    def collate_fn(self, batch):
        return self.collateBatchDicts(batch)

    def __len__(self):
        return len(self.data_items)
