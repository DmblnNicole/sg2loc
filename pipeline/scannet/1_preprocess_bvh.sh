#!/usr/bin/env bash
# Stage 1 (ScanNet): build the BVH raycasting trees for the eval map scans.
# Runs in the sg2loc conda env. Paths come from sg2loc/scannet/configs/paths.yaml.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scannet.preprocessing.generate_bvh_trees \
    --config "$REPO/sg2loc/scannet/configs/val.yaml" "$@"
