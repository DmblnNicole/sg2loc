#!/usr/bin/env bash
# Stage 2: particle-filter coarse 6DoF localization on 3RScan.
# Runs in the sg2loc conda env. Outputs go to <runs_dir>/<timestamp>/filter/.
#
# Usage:  bash pipeline/3rscan/2_particle_filter.sh [--sequence-length N]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

python -m sg2loc.scan3r.particle_filter \
    --config "$REPO/sg2loc/scan3r/configs/val.yaml" "$@"
