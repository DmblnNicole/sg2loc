#!/usr/bin/env bash
# Stage 0: download the 3RScan data needed for evaluation. Fetches the metadata
# (3RScan.json), the 3DSSG scene graphs (objects.json) and the SG2Loc provided archive
# (features, annotations, map meshes, scene centers and the SceneGraphLoc checkpoint),
# downloads the evaluation scans with the official 3RScan download script and unpacks
# their image sequences. The official script is obtained by filling out the terms-of-use
# form linked on the 3RScan project page (https://waldjohannau.github.io/RIO/). The
# other downloads are covered by the same terms of use.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RSCAN_JSON_URL="https://campar.in.tum.de/public_datasets/3RScan/3RScan.json"
OBJECTS_JSON_URL="https://campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json"
PROVIDED_URL="https://drive.usercontent.google.com/download?id=1S-lBXJV8oNErp0pf7j4A10VsThC17lhU&export=download&confirm=t"
SCENE_LISTS="$REPO/sg2loc/scan3r/preprocessing/scene_lists"

usage() {
    echo "Usage: $0 --official-script /path/to/download.py [--data-root DIR] [--all]"
    echo "  --official-script  path to the official 3RScan download script, received after"
    echo "                     filling out the terms-of-use form on the 3RScan project page"
    echo "                     (https://waldjohannau.github.io/RIO/)"
    echo "  --data-root        target data root (default: data_root_dir from"
    echo "                     sg2loc/scan3r/configs/paths.yaml)"
    echo "  --all              download the full dataset instead of the 60 evaluation scans"
    exit 1
}

OFFICIAL=""
DATA_ROOT=""
ALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --official-script) OFFICIAL="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        *) usage ;;
    esac
done
if [[ -z "$OFFICIAL" ]]; then
    usage
fi
if [[ ! -f "$OFFICIAL" ]]; then
    echo "ERROR: official download script not found at $OFFICIAL"
    exit 1
fi

if [[ -z "$DATA_ROOT" ]]; then
    DATA_ROOT="$(sed -n 's/^data_root_dir:[[:space:]]*\([^#]*\).*/\1/p' \
        "$REPO/sg2loc/scan3r/configs/paths.yaml" | sed 's/[[:space:]]*$//')"
fi
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$REPO/$DATA_ROOT"
fi

RESUME_DIR="$(sed -n 's/^resume_dir:[[:space:]]*\([^#]*\).*/\1/p' \
    "$REPO/sg2loc/scan3r/configs/paths.yaml" | sed 's/[[:space:]]*$//')"
if [[ "$RESUME_DIR" != /* ]]; then
    RESUME_DIR="$REPO/$RESUME_DIR"
fi

echo "Data root: $DATA_ROOT"
echo "Downloading requires that you have agreed to the 3RScan terms of use"
echo "(the form linked on https://waldjohannau.github.io/RIO/)."
read -r -p "Press Enter to confirm and continue, CTRL-C to abort. "

mkdir -p "$DATA_ROOT/files" "$DATA_ROOT/scenes"

if [[ -f "$DATA_ROOT/files/3RScan.json" ]]; then
    echo "files/3RScan.json exists, skipping."
else
    echo "Downloading 3RScan.json."
    curl -fL --retry 3 --progress-bar "$RSCAN_JSON_URL" -o "$DATA_ROOT/files/3RScan.json.part"
    mv "$DATA_ROOT/files/3RScan.json.part" "$DATA_ROOT/files/3RScan.json"
fi

if [[ -f "$DATA_ROOT/files/objects.json" ]]; then
    echo "files/objects.json exists, skipping."
else
    echo "Downloading 3DSSG objects.json."
    curl -fL --retry 3 --progress-bar "$OBJECTS_JSON_URL" -o "$DATA_ROOT/files/objects.json.part"
    mv "$DATA_ROOT/files/objects.json.part" "$DATA_ROOT/files/objects.json"
fi

PROVIDED_ZIP="$DATA_ROOT/sg2loc_3rscan_provided.zip"
if [[ -d "$DATA_ROOT/files/patch_anno" && -f "$RESUME_DIR/pretrained.pth.tar" ]]; then
    echo "Provided archive already unpacked, skipping."
else
    if [[ ! -f "$PROVIDED_ZIP" ]]; then
        echo "Downloading the SG2Loc provided archive (7.5 GB)."
        curl -fL --retry 3 -C - --progress-bar "$PROVIDED_URL" -o "$PROVIDED_ZIP.part"
        mv "$PROVIDED_ZIP.part" "$PROVIDED_ZIP"
    fi
    echo "Unpacking the provided archive."
    python -m zipfile -e "$PROVIDED_ZIP" "$DATA_ROOT"
    mkdir -p "$RESUME_DIR"
    mv "$DATA_ROOT/checkpoints/pretrained.pth.tar" "$RESUME_DIR/"
    rmdir "$DATA_ROOT/checkpoints"
    rm "$PROVIDED_ZIP"
    echo "Checkpoint installed at $RESUME_DIR/pretrained.pth.tar."
fi

unpack_sequence() {
    local scan_dir="$1"
    if [[ -f "$scan_dir/sequence/_info.txt" ]]; then
        return
    fi
    echo "Unpacking $(basename "$scan_dir")/sequence.zip."
    mkdir -p "$scan_dir/sequence"
    python -m zipfile -e "$scan_dir/sequence.zip" "$scan_dir/sequence"
}

if [[ $ALL -eq 1 ]]; then
    echo "Downloading the full 3RScan dataset."
    yes '' | python "$OFFICIAL" -o "$DATA_ROOT/scenes"
    for scan_dir in "$DATA_ROOT/scenes"/*/; do
        unpack_sequence "${scan_dir%/}"
    done
else
    scan_ids="$(cat "$SCENE_LISTS/scan3r_eval_query_scans.txt" \
        "$SCENE_LISTS/scan3r_eval_rescans.txt")"
    total="$(echo "$scan_ids" | wc -l)"
    echo "Downloading the $total evaluation scans."
    n=0
    for scan_id in $scan_ids; do
        n=$((n + 1))
        if [[ -f "$DATA_ROOT/scenes/$scan_id/sequence/_info.txt" ]]; then
            echo "[$n/$total] $scan_id exists, skipping."
            continue
        fi
        echo "[$n/$total] Downloading $scan_id."
        yes '' | python "$OFFICIAL" -o "$DATA_ROOT/scenes" --id "$scan_id"
        unpack_sequence "$DATA_ROOT/scenes/$scan_id"
    done
fi

echo "Done. Data root: $DATA_ROOT"
