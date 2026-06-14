<h1 align="center">SG2Loc: Sequential Visual Localization<br>on 3D Scene Graphs</h1>

This repository implements the paper [SG2Loc](https://arxiv.org/abs/2606.11880):
localizing image sequences in the 3D scene graph of a room. A semantic particle filter
over a lightweight map gives a coarse pose, and rendered-view matching refines it.

<p align="center">
  <img src="assets/sg2loc_demo.gif" width="100%" alt="SG2Loc localizing a query sequence"/>
</p>

<p align="center"><i>SG2Loc localizing a query sequence into a scene graph
(illustrative demo).</i></p>

## Installation

Clone the repository:

```bash
git clone https://github.com/DmblnNicole/sg2loc-clean.git
cd sg2loc-clean
```

Create the `sg2loc` environment (Python 3.10):

```bash
conda env create -f environment.yml
conda activate sg2loc
pip install torch==2.12.1+cu129 torchvision==0.27.1+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install --no-deps git+https://github.com/Parskatt/RoMaV2.git@7151f3846ad0c89c213afb6803966484a6dd76e0
```

## 3RScan

### 1. Download the data

Fill out the [3RScan terms-of-use](https://waldjohannau.github.io/RIO/) to receive the
official download script (a python file), then pass it to our download script:

```bash
bash pipeline/3rscan/0_download_data.sh --official-script /path/to/download.py
```

This downloads the scans, the [3DSSG](https://3dssg.github.io/) scene graphs and [SceneGraphLoc](https://github.com/y9miao/VLSG) checkpoint into default folders. For a custom data
root, pass `--data-root` to the download and run commands, or set `data_root_dir` in
`sg2loc/scan3r/configs/paths.yaml`.

### 2. Preprocess

Cache the annotated meshes and build the BVH raycasting trees (needed once):

```bash
bash pipeline/3rscan/1_preprocess_bvh.sh
```

### 3. Localize

Run the particle filter:

```bash
bash pipeline/3rscan/2_particle_filter.sh --sequence-length 5
```

Any sequence length of at least 2 works, the paper reports 5, 10 and 25. The coarse poses
and metrics land in `results/3rscan/<timestamp>/filter/`.

Refine the poses:

```bash
bash pipeline/3rscan/3_refinement.sh
```

The refinement picks up the newest particle-filter run; pass `--coarse-csv <csv>` for a
specific one. The refined poses and metrics land in
`results/3rscan/<timestamp>/refine_<timestamp>/`.

### Visualization

```bash
bash pipeline/3rscan/2_particle_filter.sh --sequence-length 10 --debug --scene 0988ea72
```

Renders one GIF per sequence into `results/3rscan/<timestamp>/viz/`.

## ScanNet

The same pipeline for ScanNet.

### 1. Download the data

Agree to the [ScanNet terms of use](https://github.com/ScanNet/ScanNet) to receive the
official download script (a python file), then pass it to our download script:

```bash
bash pipeline/scannet/0_download_data.sh --official-script /path/to/download-scannet.py
```

This downloads the 48 evaluation scan pairs, our
[provided archive](https://drive.google.com/file/d/1GpJy65Be1HJRImB_GnTZml9x9sce2NcO/view)
(~19 GB: features, SceneGraphFusion scene graphs, annotations) and
[SceneGraphLoc](https://github.com/y9miao/VLSG) checkpoint into a default folder. For a custom data
root, pass `--data-root` to the download and run commands, or set `data_root_dir` in
`sg2loc/scannet/configs/paths.yaml`.

### 2. Preprocess

Build the BVH raycasting trees (needed once):

```bash
bash pipeline/scannet/1_preprocess_bvh.sh
```

### 3. Localize

Run the particle filter:

```bash
bash pipeline/scannet/2_particle_filter.sh --sequence-length 5
```

Refine the poses:

```bash
bash pipeline/scannet/3_refinement.sh
```

## Acknowledgements

SG2Loc builds on [SceneGraphLoc](https://github.com/y9miao/VLSG) and reuses parts of its
codebase. Many thanks to its authors.

## Citation

If you find the paper or code useful, please consider citing:

```bibtex
@inproceedings{damblon2026sg2loc,
  title     = {SG2Loc: Sequential Visual Localization on 3D Scene Graphs},
  author    = {Damblon, Nicole and Vysotska, Olga and Tombari, Federico and Pollefeys, Marc and Barath, Daniel},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  eprint    = {arXiv:2606.11880},
}
```
