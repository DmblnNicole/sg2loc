"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.

Usage:
    python -m sg2loc.scannet.preprocessing.generate_2d_features \
        --config sg2loc/scannet/configs/val.yaml --scans-dir /path/to/scannet/scans \
        --files-dir /path/to/scannet/files --scene-list <scans txt>
"""

import argparse
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvf
from tqdm.auto import tqdm

from sg2loc.configs import config, update_config
from sg2loc.preprocessing.dinov2_utils import DinoV2ExtractFeatures
from sg2loc.scannet.utils import load_frame_idxs
from sg2loc.utils import common

DINO_MODEL = "dinov2_vitg14"
DESC_LAYER = 31  # transformer block the descriptors are hooked from
DESC_FACET = "value"  # value facet of the attention block
DINO_PATCH_SIZE = 14  # px per DinoV2 patch, sets the backbone input resolution
INFERENCE_BATCH = 25  # frames per forward pass


class ScannetDinov2Generator:
    def __init__(self, cfg, scans_dir: str, files_dir: str, scan_ids):
        self.scan_ids = scan_ids

        # out dir
        self.feat_2D_out_dir = osp.join(files_dir, cfg.data.img_encoding.feature_dir)
        common.ensure_dir(self.feat_2D_out_dir)

        # get image paths
        self.img_step = cfg.data.img.img_step
        self.img_paths = {}
        for scan_id in self.scan_ids:
            frame_idxs = load_frame_idxs(scans_dir, scan_id, self.img_step)
            self.img_paths[scan_id] = {
                frame_idx: osp.join(scans_dir, scan_id, "color", f"{frame_idx}.jpg")
                for frame_idx in frame_idxs
            }

        # feature inference config (backbone input size follows the patch grid, 14 px per patch)
        self.inference_step = INFERENCE_BATCH
        self.image_resize_w = cfg.data.img_encoding.patch_w * DINO_PATCH_SIZE
        self.image_resize_h = cfg.data.img_encoding.patch_h * DINO_PATCH_SIZE

    def register_model(self):
        self.device = torch.device("cuda")
        self.extractor = DinoV2ExtractFeatures(
            DINO_MODEL, DESC_LAYER, DESC_FACET, device=self.device
        )
        self.base_tf = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def generateFeatures(self):
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                imgs_features = self.generateFeaturesEachScan(scan_id)

            # save features in frame-level
            out_scan_folder = osp.join(self.feat_2D_out_dir, scan_id)
            common.ensure_dir(out_scan_folder)
            for frame_idx in imgs_features:
                out_file = osp.join(out_scan_folder, f"{frame_idx}.npy")
                np.save(out_file, imgs_features[frame_idx])

    def generateFeaturesEachScan(self, scan_id):
        imgs_features = {}
        # load images
        img_paths = self.img_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()

        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

            tensor_idxs_to_frame_idxs = {}
            img_tensors_list = []
            if len(frame_idxs_sublist) == 0:
                continue

            for idx, frame_idx in enumerate(frame_idxs_sublist):
                img_path = img_paths[frame_idx]
                img = Image.open(img_path).convert("RGB")
                h_new = (self.image_resize_h // 14) * 14
                w_new = (self.image_resize_w // 14) * 14
                img_pt = img.resize((w_new, h_new), Image.BICUBIC)

                img_pt = self.base_tf(img_pt)
                img_tensors_list.append(img_pt)
                tensor_idxs_to_frame_idxs[idx] = frame_idx
            # inference
            imgs_tensor = torch.stack(img_tensors_list, dim=0).float().to(self.device)
            ret = self.extractor(imgs_tensor)  # [num_images, num_patches, desc_dim]

            for idx, frame_idx in tensor_idxs_to_frame_idxs.items():
                imgs_features[frame_idx] = ret[idx].cpu().numpy()
        return imgs_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="backbone config (val.yaml); provides img_step, patch grid and the feature dir",
    )
    parser.add_argument("--scans-dir", required=True, help="directory with the ScanNet scans")
    parser.add_argument("--files-dir", required=True, help="the dataset files/ directory")
    parser.add_argument("--scene-list", required=True, help="txt file, one scan per line")
    parser.add_argument(
        "--data-root", default="", type=str, help="dataset root, overrides paths.yaml"
    )
    args = parser.parse_args()

    cfg = update_config(config, args.config, ensure_dir=False, data_root=args.data_root)
    scan_ids = [ln.strip() for ln in open(args.scene_list) if ln.strip()]
    generator = ScannetDinov2Generator(cfg, args.scans_dir, args.files_dir, scan_ids)
    generator.register_model()
    generator.generateFeatures()


if __name__ == "__main__":
    main()
