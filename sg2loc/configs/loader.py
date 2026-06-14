"""
Loads the SG2Loc config: merges val.yaml with the sibling sg2loc.yaml,
resolves the machine paths from paths.yaml and freezes the result.
"""

import os
import os.path as osp
import shutil

import yaml
from yacs.config import CfgNode as CN

from sg2loc.utils import common

_SG2LOC_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
_REPO_DIR = osp.dirname(_SG2LOC_DIR)
_CONFIG_DIR = osp.join(_SG2LOC_DIR, "configs")

# machine-path keys in paths.yaml and the environment variables that override them
_PATHS_ENV = {
    "data_root_dir": "Data_ROOT_DIR",
    "resume_dir": "RESUME_DIR",
    "runs_dir": "ROOM_RETRIEVAL_OUT_DIR",
}
_PATHS_DEFAULTS = {"runs_dir": "results"}

_C = CN()

# dataset
_C.data = CN()
_C.data.root_dir = ""
_C.data.rescan = False
_C.data.temporal = False
_C.data.resplit = False

_C.data.img = CN()
_C.data.img.img_step = 5
_C.data.img.w = 960
_C.data.img.h = 540

_C.data.img_encoding = CN()
_C.data.img_encoding.resize_w = 1024
_C.data.img_encoding.img_rotate = True  # rotate w,h for backbone GCVit
_C.data.img_encoding.patch_w = 16  # number of patches in width
_C.data.img_encoding.patch_h = 9
_C.data.img_encoding.record_feature = False
_C.data.img_encoding.use_feature = False
_C.data.img_encoding.preload_feature = False
_C.data.img_encoding.feature_dir = ""

_C.data.cross_scene = CN()
_C.data.cross_scene.use_cross_scene = False
_C.data.cross_scene.num_scenes = 0
_C.data.cross_scene.num_negative_samples = 0
_C.data.cross_scene.use_tf_idf = False

_C.data.scene_graph = CN()
_C.data.scene_graph.obj_img_patch = ""
_C.data.scene_graph.obj_patch_num = 1
_C.data.scene_graph.obj_topk = 1

# model
_C.model = CN()
_C.model.backbone = CN()
_C.model.backbone.cfg_file = ""
_C.model.backbone.pretrained = ""
_C.model.backbone.num_reduce = 1
_C.model.backbone.backbone_dim = 512
_C.model.patch = CN()
_C.model.patch.hidden_dims = []
_C.model.patch.encoder_dim = 256
_C.model.patch.gcn_layers = 0
_C.model.obj = CN()
_C.model.obj.embedding_dim = 256
_C.model.obj.embedding_hidden_dims = []
_C.model.obj.encoder_dim = 256
_C.model.other = CN()
_C.model.other.drop = 0.0
_C.model.global_descriptor_dim = 1024

# sgaligner
_C.sgaligner = CN()
_C.sgaligner.modules = ["point", "gat", "rel", "attr"]
_C.sgaligner.use_pos_enc = False

# train-section keys the dataset and model constructors read
_C.train = CN()
_C.train.loss = CN()
_C.train.loss.use_temporal = False
_C.train.loss.use_global_descriptor = False
_C.train.data_aug = CN()
_C.train.data_aug.use_aug = False
_C.train.data_aug.img = CN()
_C.train.data_aug.img.rotation = 0.0
_C.train.data_aug.img.horizontal_flip = 0.0
_C.train.data_aug.img.vertical_flip = 0.0
_C.train.data_aug.img.color = 0.0
_C.train.data_aug.use_aug_3D = False
_C.train.data_aug.pcs = CN()
_C.train.data_aug.pcs.granularity = [0.05]
_C.train.data_aug.pcs.magnitude = [0.0]

# validation
_C.val = CN()
_C.val.num_workers = 1
_C.val.room_retrieval = CN()
_C.val.room_retrieval.epsilon_th = 0.8
_C.val.room_retrieval.method_name = ""

# others
_C.other = CN()
_C.other.use_resume = False
_C.other.resume = ""


def machine_paths(config_dir: str = _CONFIG_DIR) -> dict:
    """Resolve the machine paths from paths.yaml."""
    values = dict(_PATHS_DEFAULTS)
    paths_file = osp.join(config_dir, "paths.yaml")
    if osp.exists(paths_file):
        with open(paths_file) as f:
            values.update(yaml.safe_load(f) or {})
    resolved = {}
    for key, env_name in _PATHS_ENV.items():
        value = os.getenv(env_name) or values.get(key)
        if value:
            resolved[key] = osp.join(_REPO_DIR, osp.expanduser(str(value)))
    return resolved


def require_path(paths: dict, key: str) -> str:
    if key not in paths:
        raise OSError(
            f"machine path '{key}' is not set. Set it in paths.yaml next to the config, "
            f"or set the {_PATHS_ENV[key]} environment variable."
        )
    # runs_dir is created on demand, the input dirs must already exist
    if key != "runs_dir" and not osp.isdir(paths[key]):
        raise OSError(
            f"machine path '{key}' points to a non-existing directory: {paths[key]}. "
            "Set it in paths.yaml next to the config."
        )
    return paths[key]


def snapshot_configs(config_path: str, out_dir: str, sequence_length: int) -> None:
    """Snapshot val.yaml and sg2loc.yaml into the run dir with the sequence length."""
    shutil.copy(config_path, osp.join(out_dir, "val.yaml"))
    overlay = osp.join(osp.dirname(config_path), "sg2loc.yaml")
    if not osp.exists(overlay):
        return
    with open(overlay) as f:
        data = yaml.safe_load(f)
    # the refinement reads this value back, so record a --sequence-length override too
    data["particle_filter"]["sequence_length"] = sequence_length
    with open(osp.join(out_dir, "sg2loc.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def update_config(cfg, filename, ensure_dir=True, data_root=""):
    """Merge val.yaml with the sibling sg2loc.yaml overlay, resolve machine paths and freeze.

    data_root overrides the data_root_dir from paths.yaml when given.
    """
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(filename)

    # SG2Loc method parameters live in a dedicated sg2loc.yaml
    overlay = osp.join(osp.dirname(filename), "sg2loc.yaml")
    if osp.exists(overlay):
        cfg.merge_from_file(overlay)

    paths = machine_paths(osp.dirname(osp.abspath(filename)))
    if data_root:
        paths["data_root_dir"] = osp.join(_REPO_DIR, osp.expanduser(data_root))
    cfg.data.root_dir = require_path(paths, "data_root_dir")

    # particle-filter data paths are relative to Data_ROOT_DIR
    if hasattr(cfg, "particle_filter"):
        cfg.particle_filter.scans_scenes_dir = osp.join(
            cfg.data.root_dir, cfg.particle_filter.scans_scenes_dir
        )
        cfg.particle_filter.preprocess.output_dir = osp.join(
            cfg.data.root_dir, cfg.particle_filter.preprocess.output_dir
        )
        cfg.particle_filter.preprocess.embeddings_dir = osp.join(
            cfg.data.root_dir, cfg.particle_filter.preprocess.embeddings_dir
        )

    if ensure_dir:
        cfg.model.backbone.cfg_file = osp.join(_SG2LOC_DIR, cfg.model.backbone.cfg_file)
        cfg.model.backbone.pretrained = osp.join(_SG2LOC_DIR, cfg.model.backbone.pretrained)
        cfg.output_dir = require_path(paths, "runs_dir")
        cfg.other.resume = osp.join(require_path(paths, "resume_dir"), cfg.other.resume)
        common.ensure_dir(osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name))

    cfg.freeze()
    return cfg
