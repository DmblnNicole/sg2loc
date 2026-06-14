"""
Precompute the 3RScan per-scene object embeddings (see preprocessing/object_embeddings.py).

Usage:
    python -m sg2loc.scan3r.preprocessing.precompute_object_embeddings \
        --config sg2loc/scan3r/configs/val.yaml
"""

from sg2loc.preprocessing.object_embeddings import precompute_main
from sg2loc.scan3r.dataset import EVAL_SCANS_FILE, PatchObjectPairXTAESGIDataSet
from sg2loc.scan3r.particle_filter import Scan3RParticleFilter

if __name__ == "__main__":
    precompute_main(PatchObjectPairXTAESGIDataSet, Scan3RParticleFilter, EVAL_SCANS_FILE)
