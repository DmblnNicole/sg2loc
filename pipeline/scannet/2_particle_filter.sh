#!/usr/bin/env bash
# Stage 2 (ScanNet): particle-filter coarse 6DoF localization.
# Runs in the sg2loc conda env. Outputs go to <runs_dir>/<timestamp>/filter/.
#
# Usage:  bash pipeline/scannet/2_particle_filter.sh [--sequence-length N]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scannet.particle_filter \
    --config "$REPO/sg2loc/scannet/configs/val.yaml" "$@"
