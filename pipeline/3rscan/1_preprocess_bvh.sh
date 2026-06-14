#!/usr/bin/env bash
# Stage 1: cache the annotated meshes and build the BVH raycasting trees.
# Runs in the sg2loc conda env. Paths come from sg2loc/scan3r/configs/paths.yaml.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scan3r.preprocessing.align_annotated_ply \
    --config "$REPO/sg2loc/scan3r/configs/val.yaml" "$@"

python -m sg2loc.scan3r.preprocessing.annotated_ply_to_npy \
    --config "$REPO/sg2loc/scan3r/configs/val.yaml" "$@"

python -m sg2loc.scan3r.preprocessing.generate_bvh_trees \
    --config "$REPO/sg2loc/scan3r/configs/val.yaml" \
    --scan-list "$REPO/sg2loc/scan3r/preprocessing/scene_lists/scan3r_eval_rescans.txt" "$@"
