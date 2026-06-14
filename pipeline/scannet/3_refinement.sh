#!/usr/bin/env bash
# Stage 3 (ScanNet): pose refinement of the particle-filter poses.
# Runs in the sg2loc conda env (romav2 + poselib). Refines the latest
# particle-filter run under <runs_dir> (or the CSV passed with --coarse-csv) and writes
# into the same runs folder: <runs_dir>/<ts>/refine_<ts>/.
#
# Usage:  bash pipeline/scannet/3_refinement.sh [--coarse-csv /path/to/filter/sequence_poses_and_errors.csv]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scannet.refinement \
    --config "$REPO/sg2loc/scannet/configs/val.yaml" "$@"
