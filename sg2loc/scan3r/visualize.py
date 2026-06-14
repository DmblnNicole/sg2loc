"""Render particle-filter debug snapshots into per-sequence GIFs, used by the runner with --debug."""

from __future__ import annotations

import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R

from sg2loc.particle_filter.raycasting import RayCasting
from sg2loc.particle_filter.scene import load_scene_geometry
from sg2loc.scan3r import utils as scan3r
from sg2loc.scan3r.utils import load_rescan_transforms

CMAP = plt.get_cmap("plasma")  # dark blue = low weight, bright yellow = high weight
LOW_WEIGHT_COLOR = CMAP(0.15)
ARROW_LEN_M = 0.55
PARTICLE_ARROW_LEN_M = 0.18
MAX_DRAWN_PARTICLES = 1500
GIF_WIDTH = 1400
GIF_FRAME_MS = 500


def yaw_of(rot: np.ndarray) -> float:
    xa = rot[:, 0]
    ref = np.array([1.0, 0.0, 0.0])
    ref_ip = ref - np.dot(xa, ref) * xa
    ref_ip /= np.linalg.norm(ref_ip)
    bit = np.cross(xa, ref_ip)
    return float(np.arctan2(np.dot(rot[:, 1], bit), np.dot(rot[:, 1], ref_ip)))


class SequenceRenderer:
    _scene_cache: dict = {}

    def __init__(self, cfg, scan_id: str, target_scan: str):
        self.cfg = cfg
        self.scenes_dir = osp.join(cfg.data.root_dir, "scenes")
        self.scan_id = scan_id
        self.target = target_scan
        tf = load_rescan_transforms(osp.join(cfg.data.root_dir, "files/3RScan.json"))
        self.scan_to_rescan = np.linalg.inv(tf[self.target]["transform_matrix"])
        self._manifolds: dict = {}

        if self.target not in SequenceRenderer._scene_cache:
            bvh_dir = osp.join(
                cfg.data.root_dir, cfg.particle_filter.preprocess.output_dir, self.target
            )
            SequenceRenderer._scene_cache[self.target] = load_scene_geometry(bvh_dir)
        intrinsics = scan3r.load_intrinsics(self.scenes_dir, self.scan_id)["intrinsic_mat"]
        self.raycaster = RayCasting(cfg, intrinsics, stride=1)
        self.raycaster.set_scene(SequenceRenderer._scene_cache[self.target])
        self.texture = np.asarray(
            Image.open(osp.join(self.scenes_dir, self.target, "mesh.refined_0.png")).convert("RGB")
        )

        ply = PlyData.read(
            osp.join(self.scenes_dir, self.target, "labels.instances.annotated.v2.ply")
        )
        v = ply["vertex"]
        pts = np.column_stack([v["x"], v["y"], v["z"]])
        sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 120000), replace=False)
        self.map_pts = pts[sel]
        # percentile bounds so stray points outside the room do not shrink the map
        m0 = np.percentile(self.map_pts[:, :2], 1, axis=0)
        m1 = np.percentile(self.map_pts[:, :2], 99, axis=0)
        center, half = (m0 + m1) / 2, (m1 - m0).max() / 2 + 0.4
        self.xlim = (center[0] - half, center[0] + half)
        self.ylim = (center[1] - half, center[1] + half)

    def manifold(self, frame_id: str) -> tuple:
        if frame_id not in self._manifolds:
            pose = (
                self.scan_to_rescan
                @ scan3r.load_all_poses(self.scenes_dir, self.scan_id, [frame_id])[0]
            )
            x = pose[:3, 0] / np.linalg.norm(pose[:3, 0])
            y = np.array([0.0, 1.0, 0.0])
            y = y - np.dot(y, x) * x
            y /= np.linalg.norm(y)
            fixed = np.column_stack((x, y, np.cross(x, y)))
            self._manifolds[frame_id] = (x, fixed, yaw_of(fixed))
        return self._manifolds[frame_id]

    def pose_from(self, xyz: np.ndarray, yaw: float, frame_id: str) -> np.ndarray:
        x, fixed, yaw_fixed = self.manifold(frame_id)
        pose = np.eye(4)
        pose[:3, :3] = R.from_rotvec((yaw - yaw_fixed) * x).as_matrix() @ fixed
        pose[:3, 3] = xyz
        return pose

    def view_dirs_xy(self, yaws: np.ndarray, frame_id: str) -> np.ndarray:
        """Unit xy viewing directions for a batch of particle yaws."""
        x, fixed, yaw_fixed = self.manifold(frame_id)
        rots = R.from_rotvec(np.outer(yaws - yaw_fixed, x)).as_matrix() @ fixed
        v = rots[:, :2, 2]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def render_mle(self, xyz: np.ndarray, yaw: float, frame_id: str) -> np.ndarray:
        rot = self.pose_from(xyz, yaw, frame_id)[:3, :3]
        uv, _, depth = self.raycaster.cast_depth_uv(
            rot[None], np.asarray(xyz, dtype=np.float64)[None]
        )
        h, w = self.cfg.data.img.h, self.cfg.data.img.w
        uv = np.asarray(uv).reshape(-1, 2)
        uv[:, 1] = 1.0 - uv[:, 1]
        th, tw, _ = self.texture.shape
        px = np.clip((uv * [tw - 1, th - 1]).astype(int), 0, [tw - 1, th - 1])
        img = self.texture[px[:, 1], px[:, 0]].reshape(h, w, 3).copy()
        img[np.asarray(depth).reshape(h, w) > 999] = 30  # rays that miss the mesh
        return np.rot90(img, k=-1)  # stored landscape, content is portrait

    def query_image(self, frame_id: str) -> np.ndarray:
        p = osp.join(self.scenes_dir, self.scan_id, "sequence", f"frame-{frame_id}.color.jpg")
        return np.rot90(np.asarray(Image.open(p)), k=-1)


def _pose_marker(ax, x: float, y: float, view_vec: np.ndarray, color: str, label: str) -> None:
    v = np.arctan2(view_vec[1], view_vec[0])
    ax.plot(x, y, "o", color=color, ms=16, mec="black", mew=1.3, zorder=6, label=label)
    ax.arrow(
        x,
        y,
        ARROW_LEN_M * np.cos(v),
        ARROW_LEN_M * np.sin(v),
        head_width=0.16,
        width=0.07,
        color=color,
        ec="black",
        lw=0.5,
        zorder=6,
    )


def render_frame(ctx: SequenceRenderer, d, mle_state, mle_render) -> Image.Image:
    fig = plt.figure(figsize=(16.0, 8.4), dpi=125)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.0, 0.78, 0.78],
        wspace=0.02,
        left=0.005,
        right=0.995,
        top=0.955,
        bottom=0.01,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_query = fig.add_subplot(gs[0, 1])
    ax_render = fig.add_subplot(gs[0, 2])

    ax.scatter(
        ctx.map_pts[:, 0],
        ctx.map_pts[:, 1],
        s=0.3,
        c=ctx.map_pts[:, 2],
        cmap="gray",
        alpha=0.15,
        zorder=1,
    )
    xyz, w = d["xyz"], d["weights"]
    frame_id = str(d["frame_id"])
    # cap the drawn particles so dense clouds stay readable as arrows
    if len(xyz) > MAX_DRAWN_PARTICLES:
        keep = np.random.default_rng(0).choice(len(xyz), MAX_DRAWN_PARTICLES, replace=False)
        xyz, w = xyz[keep], w[keep]
        yaws = np.asarray(d["yaw"], dtype=np.float64)[keep]
    else:
        yaws = np.asarray(d["yaw"], dtype=np.float64)
    dirs = ctx.view_dirs_xy(yaws, frame_id)
    if str(d["tag"]) == "update" and w.max() > 0:
        # draw low weights first so the warm high-weight arrows stay on top. Color by
        # weight rank, the raw weights span too many decades to read on a fixed scale.
        order = np.argsort(w)
        xyz, dirs = xyz[order], dirs[order]
        colors = CMAP(np.linspace(0.0, 1.0, len(w)))
        alpha = 0.85
    else:
        # weights are uniform outside update steps: show all particles as low-weight blue
        colors = LOW_WEIGHT_COLOR
        alpha = 0.8
    ax.quiver(
        xyz[:, 0],
        xyz[:, 1],
        dirs[:, 0] * PARTICLE_ARROW_LEN_M,
        dirs[:, 1] * PARTICLE_ARROW_LEN_M,
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0035,
        headwidth=3.4,
        headlength=3.8,
        headaxislength=3.4,
        alpha=alpha,
        zorder=3,
    )

    gt = d["gt"]
    gt_view = ctx.pose_from(gt[:3], float(gt[3]), frame_id)[:3, 2]
    _pose_marker(ax, gt[0], gt[1], gt_view, "#00ff00", "GT")
    info = f"pass {int(d['num_pass'])}   frame {frame_id}"
    if mle_state is not None:
        mle_view = ctx.pose_from(mle_state[:3], float(mle_state[3]), frame_id)[:3, 2]
        _pose_marker(ax, mle_state[0], mle_state[1], mle_view, "#ff0000", "MLE")
        pos_err = float(np.linalg.norm(mle_state[:3] - gt[:3]))
        rot_err = abs(np.degrees((mle_state[3] - gt[3] + np.pi) % (2 * np.pi) - np.pi))
        info += f"\nMLE error  {pos_err:.2f} m   {rot_err:.1f}°"
    ax.text(
        0.015,
        0.985,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        family="monospace",
        zorder=7,
        bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 3},
    )
    ax.legend(loc="upper right", fontsize=12, framealpha=0.75)
    ax.set_xlim(*ctx.xlim)
    ax.set_ylim(*ctx.ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    # frame 0 is never localized, so the image panels stay empty until the first update
    ax_query.axis("off")
    ax_render.axis("off")
    if mle_render is not None:
        ax_query.imshow(ctx.query_image(frame_id))
        ax_query.set_title("query image", fontsize=12, pad=3)
        ax_render.imshow(mle_render)
        ax_render.set_title("render from MLE", fontsize=12, pad=3)

    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return frame.resize((GIF_WIDTH, int(frame.height * GIF_WIDTH / frame.width)))


def write_gif(frames_rgb: list, gif_path: str) -> None:
    # one shared palette for the whole GIF so marker colors stay identical per frame.
    # The marker colors and the weight colormap get reserved palette slots, otherwise
    # the photo panels dominate the quantization and wash the arrows out.
    sample = frames_rgb[:: max(1, len(frames_rgb) // 6)][:6]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * im.height))
    reserved = [(255, 0, 0), (0, 255, 0), (200, 0, 0), (0, 200, 0), (0, 0, 0), (255, 255, 255)]
    reserved += [tuple(int(c * 255) for c in CMAP(t)[:3]) for t in np.linspace(0.0, 1.0, 48)]
    n_quant = 256 - len(reserved)
    palette = strip.quantize(colors=n_quant)
    pal = palette.getpalette()[: n_quant * 3]
    for rgb in reserved:
        pal += list(rgb)
    palette.putpalette(pal)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames_rgb]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
        optimize=True,
    )


def render_sequence_gif(
    cfg, scan_id: str, target_scan: str, snapshots: list, gif_path: str
) -> None:
    ctx = SequenceRenderer(cfg, scan_id, target_scan)
    mle_state, mle_render = None, None
    frames = []
    for d in snapshots:
        # the MLE only means anything right after a weight update. Carry it through the
        # resample of the same frame. Frame 0 gets no update, so pass starts and
        # reinits show no estimate instead of the previous frame's MLE.
        tag = str(d["tag"])
        if tag in ("start", "reinit"):
            mle_state, mle_render = None, None
        elif tag == "update":
            mle_state = d["mle"]
            mle_render = ctx.render_mle(mle_state[:3], float(mle_state[3]), str(d["frame_id"]))
        frames.append(render_frame(ctx, d, mle_state, mle_render))
    write_gif(frames, gif_path)
