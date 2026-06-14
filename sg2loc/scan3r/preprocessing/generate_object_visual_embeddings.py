"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.

Usage:
    python -m sg2loc.scan3r.preprocessing.generate_object_visual_embeddings \
        --config sg2loc/scan3r/configs/val.yaml --scans-dir /path/to/3RScan/scenes \
        --files-dir /path/to/3RScan/files --scene-list <scans txt>
"""

import argparse
import os.path as osp

import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision import transforms as tvf

from sg2loc.configs import config, update_config
from sg2loc.preprocessing.dinov2_utils import DinoV2ExtractFeatures
from sg2loc.scan3r import utils as scan3r
from sg2loc.utils import common

DINO_MODEL = "dinov2_vitg14"
DESC_LAYER = 31  # transformer block the descriptors are hooked from
DESC_FACET = "value"  # value facet of the attention block
# per-pixel object-id annotation (raycast of the annotated mesh), relative to files/
GT_ANNO_DIR = "gt_projection/obj_id_pkl"

# multiview config
multi_level_expansion_ratio = 0.2
num_of_levels = 3


# openmask3d multi-level functions
def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1


def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2, y2 = mask2box(mask)
    if level == 0:
        return x1, y1, x2, y2
    shape = mask.shape
    x_exp = int(abs(x2 - x1) * expansion_ratio) * level
    y_exp = int(abs(y2 - y1) * expansion_ratio) * level
    return (
        max(0, x1 - x_exp),
        max(0, y1 - y_exp),
        min(shape[1], x2 + x_exp),
        min(shape[0], y2 + y_exp),
    )


class ObjVisualEmbGen:
    def __init__(self, cfg, scans_dir: str, files_dir: str, scan_ids):
        self.scan_ids = scan_ids

        # undefined patch anno id
        self.undefined = 0

        # 2D images
        self.img_rotate = cfg.data.img_encoding.img_rotate

        ## image patches
        self.image_path = {}
        for scan_id in self.scan_ids:
            frame_idxs = scan3r.load_frame_idxs(scans_dir, scan_id)
            self.image_path[scan_id] = {
                frame_idx: osp.join(scans_dir, scan_id, "sequence", f"frame-{frame_idx}.color.jpg")
                for frame_idx in frame_idxs
            }

        # load 2D gt obj id annotation
        self.gt_2D_anno_folder = osp.join(files_dir, GT_ANNO_DIR)
        self.obj_2D_annos_pathes = {}
        for scan_id in self.scan_ids:
            anno_2D_file = osp.join(self.gt_2D_anno_folder, f"{scan_id}.pkl")
            self.obj_2D_annos_pathes[scan_id] = anno_2D_file

        # obj visual emb config
        self.topk = cfg.data.scene_graph.obj_topk

        # out obj visual emb dir
        self.obj_visual_emb_dir = osp.join(files_dir, cfg.data.scene_graph.obj_img_patch)
        common.ensure_dir(self.obj_visual_emb_dir)

        # Dinov2 extractor
        self.device = torch.device("cuda")
        self.extractor = DinoV2ExtractFeatures(
            DINO_MODEL, DESC_LAYER, DESC_FACET, use_cls=True, device=self.device
        )

        self.base_tf = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def generateObjVisualEmb(self):
        for scan_id in tqdm.tqdm(self.scan_ids):
            obj_visual_emb_file = osp.join(self.obj_visual_emb_dir, f"{scan_id}.pkl")
            if osp.isfile(obj_visual_emb_file):
                continue
            obj_patch_info = self.generateObjVisualEmbScan(scan_id)
            common.write_pkl_data(obj_patch_info, obj_visual_emb_file)

    def generateObjVisualEmbScan(self, scan_id):
        obj_image_votes = {}

        # load gt 2D obj anno
        obj_anno_2D_file = self.obj_2D_annos_pathes[scan_id]
        obj_anno_2D = common.load_pkl_data(obj_anno_2D_file)

        # iterate over all frames
        for frame_idx in obj_anno_2D:
            obj_2D_anno_frame = obj_anno_2D[frame_idx]
            ## process 2D anno
            obj_ids, counts = np.unique(obj_2D_anno_frame, return_counts=True)
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                count = counts[idx]
                if obj_id == self.undefined:
                    continue
                if obj_id not in obj_image_votes:
                    obj_image_votes[obj_id] = {}
                if frame_idx not in obj_image_votes[obj_id]:
                    obj_image_votes[obj_id][frame_idx] = 0
                obj_image_votes[obj_id][frame_idx] = count
        ## select top K frames for each obj
        obj_image_votes_topK = {}
        for obj_id in obj_image_votes:
            obj_image_votes_topK[obj_id] = []
            obj_image_votes_f = obj_image_votes[obj_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.topk:
                obj_image_votes_topK[obj_id] = sorted_frame_idxs[: self.topk]
            else:
                obj_image_votes_topK[obj_id] = sorted_frame_idxs
        ## get obj visual emb
        obj_visual_emb = {}
        for obj_id in obj_image_votes_topK:
            obj_image_votes_topK_frames = obj_image_votes_topK[obj_id]
            obj_visual_emb[obj_id] = {}
            for frame_idx in obj_image_votes_topK_frames:
                obj_visual_emb[obj_id][frame_idx] = self.generate_visual_emb(
                    scan_id, frame_idx, obj_id, obj_anno_2D[frame_idx]
                )
        obj_patch_info = {
            "obj_visual_emb": obj_visual_emb,
            "obj_image_votes_topK": obj_image_votes_topK,
        }
        return obj_patch_info

    def generate_visual_emb(self, scan_id, frame_idx, obj_id, gt_anno):
        if self.img_rotate:
            obj_2D_anno_f_rot = gt_anno.transpose(1, 0)
            obj_2D_anno_f_rot = np.flip(obj_2D_anno_f_rot, 1)
        # load image
        image_path = self.image_path[scan_id][frame_idx]
        image = Image.open(image_path)
        if self.img_rotate:
            image = image.transpose(Image.ROTATE_270)
        # get obj mask
        obj_mask = obj_2D_anno_f_rot == obj_id
        # extract multi-level crop dinov2 features
        images_crops = []
        for level in range(num_of_levels):
            mask_tensor = torch.from_numpy(obj_mask).float()
            x1, y1, x2, y2 = mask2box_multi_level(mask_tensor, level, multi_level_expansion_ratio)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((224, 224), Image.BICUBIC)
            img_pt = self.base_tf(cropped_img)
            images_crops.append(img_pt)
        if len(images_crops) > 0:
            image_input = torch.stack(images_crops).to(self.device)
            with torch.no_grad():
                ret = self.extractor(image_input)  # [num_levels, 1+num_patches, desc_dim]
                # get cls token
                cls_token = ret[:, 0, :]
                # get mean of all patches
                mean_patch = cls_token.mean(dim=0)
        return mean_patch.cpu().detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); provides img_rotate, obj_topk and the output name",
    )
    parser.add_argument("--scans-dir", required=True, help="the 3RScan scenes/ directory")
    parser.add_argument("--files-dir", required=True, help="the dataset files/ directory")
    parser.add_argument("--scene-list", required=True, help="txt file, one scan per line")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    scan_ids = [ln.strip() for ln in open(args.scene_list) if ln.strip()]
    generator = ObjVisualEmbGen(cfg, args.scans_dir, args.files_dir, scan_ids)
    generator.generateObjVisualEmb()


if __name__ == "__main__":
    main()
