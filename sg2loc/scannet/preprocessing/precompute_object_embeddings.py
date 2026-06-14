"""
Precompute the ScanNet per-scene object embeddings (see preprocessing/object_embeddings.py).

Usage:
    python -m sg2loc.scannet.preprocessing.precompute_object_embeddings \
        --config sg2loc/scannet/configs/val.yaml
"""

from sg2loc.preprocessing.object_embeddings import precompute_main
from sg2loc.scannet.dataset import EVAL_SCANS_FILE, ScannetPatchObjDataset
from sg2loc.scannet.particle_filter import ScannetParticleFilter

if __name__ == "__main__":
    precompute_main(ScannetPatchObjDataset, ScannetParticleFilter, EVAL_SCANS_FILE)
