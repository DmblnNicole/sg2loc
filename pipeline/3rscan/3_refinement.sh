#!/usr/bin/env bash
# Stage 3: pose refinement of the particle-filter poses.
# Runs in the sg2loc conda env (romav2 + poselib). Refines the latest particle filter run under <runs_dir> (or the CSV passed
# with --coarse-csv) and writes into the same runs folder: <runs_dir>/<ts>/refine_<ts>/.
#
# Usage:  bash pipeline/3rscan/3_refinement.sh [--coarse-csv /path/to/filter/sequence_poses_and_errors.csv]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scan3r.refinement \
    --config "$REPO/sg2loc/scan3r/configs/val.yaml" "$@"
