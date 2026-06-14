#!/usr/bin/env bash
# Stage 0 (ScanNet): download the ScanNet data needed for evaluation. Downloads the .sens
# files of the 48 evaluation query scans and extracts their color, depth, poses and
# intrinsics, downloads the map meshes of their _01 twin scans, and downloads the SG2Loc
# provided archive (DinoV2 features, SceneGraphFusion predictions, patch annotations,
# object embeddings and the checkpoint; derived from ScanNet, so the same terms of use
# apply). The official download script is obtained by agreeing to the ScanNet terms of use
# (https://github.com/ScanNet/ScanNet).

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO"

PROVIDED_URL="https://drive.usercontent.google.com/download?id=1GpJy65Be1HJRImB_GnTZml9x9sce2NcO&export=download&confirm=t"
SCENE_LIST="$REPO/sg2loc/scannet/preprocessing/scene_lists/scannet_eval_query_scans.txt"

usage() {
    echo "Usage: $0 --official-script /path/to/download-scannet.py [--data-root DIR]"
    echo "  --official-script  path to the official ScanNet download script, received after"
    echo "                     agreeing to the ScanNet terms of use"
    echo "                     (https://github.com/ScanNet/ScanNet)"
    echo "  --data-root        target data root (default: data_root_dir from"
    echo "                     sg2loc/scannet/configs/paths.yaml)"
    exit 1
}

OFFICIAL=""
DATA_ROOT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --official-script) OFFICIAL="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
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
        "$REPO/sg2loc/scannet/configs/paths.yaml" | sed 's/[[:space:]]*$//')"
fi
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$REPO/$DATA_ROOT"
fi

RESUME_DIR="$(sed -n 's/^resume_dir:[[:space:]]*\([^#]*\).*/\1/p' \
    "$REPO/sg2loc/scannet/configs/paths.yaml" | sed 's/[[:space:]]*$//')"
if [[ "$RESUME_DIR" != /* ]]; then
    RESUME_DIR="$REPO/$RESUME_DIR"
fi

echo "Data root: $DATA_ROOT"
echo "Downloading requires that you have agreed to the ScanNet terms of use"
echo "(https://github.com/ScanNet/ScanNet)."
read -r -p "Press Enter to confirm and continue, CTRL-C to abort. "

mkdir -p "$DATA_ROOT/scans"

PROVIDED_ZIP="$DATA_ROOT/sg2loc_scannet_provided.zip"
if [[ -d "$DATA_ROOT/files/gt_patch_anno" && -f "$RESUME_DIR/epoch-31.pth.tar" ]]; then
    echo "Provided archive already unpacked, skipping."
else
    if [[ ! -f "$PROVIDED_ZIP" ]]; then
        echo "Downloading the SG2Loc provided archive (19 GB)."
        curl -fL --retry 3 -C - --progress-bar "$PROVIDED_URL" -o "$PROVIDED_ZIP.part"
        mv "$PROVIDED_ZIP.part" "$PROVIDED_ZIP"
    fi
    echo "Unpacking the provided archive."
    python -m zipfile -e "$PROVIDED_ZIP" "$DATA_ROOT"
    mkdir -p "$RESUME_DIR"
    mv "$DATA_ROOT/checkpoints/epoch-31.pth.tar" "$RESUME_DIR/"
    rmdir "$DATA_ROOT/checkpoints"
    rm "$PROVIDED_ZIP"
    echo "Checkpoint installed at $RESUME_DIR/epoch-31.pth.tar."
fi

query_scans="$(grep -v '^\s*$' "$SCENE_LIST")"

for scan_id in $query_scans; do
    scan_dir="$DATA_ROOT/scans/$scan_id"
    if compgen -G "$scan_dir/color/*.jpg" > /dev/null && [[ -d "$scan_dir/pose" ]]; then
        echo "$scan_id frames exist, skipping."
        continue
    fi
    if [[ ! -f "$scan_dir/$scan_id.sens" ]]; then
        echo "Downloading $scan_id.sens."
        yes '' | python "$OFFICIAL" -o "$DATA_ROOT" --id "$scan_id" --type .sens
    fi
    echo "Extracting $scan_id.sens."
    python -m sg2loc.scannet.preprocessing.extract_sens "$scan_dir/$scan_id.sens" "$scan_dir"
    rm "$scan_dir/$scan_id.sens"
done

# each map scan contributes its mesh and its camera intrinsics. The intrinsics sit in
# the first bytes of the map .sens, so only that header is fetched, not the whole file.
SENS_URL="https://kaldir.vc.cit.tum.de/scannet/v1/scans"

for scan_id in $query_scans; do
    map_id="${scan_id/_00/_01}"
    map_dir="$DATA_ROOT/scans/$map_id"
    ply="$map_dir/${map_id}_vh_clean_2.ply"
    if [[ -f "$ply" && -f "$map_dir/intrinsic/intrinsic_color.txt" ]]; then
        echo "$map_id map data exists, skipping."
        continue
    fi
    if [[ ! -f "$ply" ]]; then
        echo "Downloading $map_id map mesh."
        yes '' | python "$OFFICIAL" -o "$DATA_ROOT" --id "$map_id" --type _vh_clean_2.ply
    fi
    if [[ ! -f "$map_dir/intrinsic/intrinsic_color.txt" ]]; then
        echo "Downloading $map_id intrinsics (.sens header)."
        curl -fsL "$SENS_URL/$map_id/$map_id.sens" 2>/dev/null | head -c 4096 \
            > "$map_dir/$map_id.sens_header"
        python -m sg2loc.scannet.preprocessing.extract_sens --intrinsics-only \
            "$map_dir/$map_id.sens_header" "$map_dir"
        rm "$map_dir/$map_id.sens_header"
    fi
done

echo "Done. Data root: $DATA_ROOT"
