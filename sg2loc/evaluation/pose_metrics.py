"""
Calculate metrics from pose errors.
"""

from __future__ import annotations

import csv
import statistics

POS_THRESH_M = 0.25
ROT_THRESH_DEG = 2.0


def compute(errors) -> dict | None:
    """Compute pose metrics from (position_error_m, rotation_error_deg) pairs."""
    errors = list(errors)
    n = len(errors)
    if n == 0:
        return None
    pos = [p for p, _ in errors]
    rot = [r for _, r in errors]
    return {
        "n": n,
        "median_pos": statistics.median(pos),
        "median_rot": statistics.median(rot),
        "recall_pos": 100.0 * sum(p <= POS_THRESH_M for p in pos) / n,
        "recall_rot": 100.0 * sum(r <= ROT_THRESH_DEG for r in rot) / n,
        "recall_joint": 100.0
        * sum(p <= POS_THRESH_M and r <= ROT_THRESH_DEG for p, r in errors)
        / n,
    }


def read_sequence_errors(sequence_csv: str) -> list:
    out = []
    with open(sequence_csv) as f:
        for row in csv.DictReader(f):
            out.append((float(row["PositionError"]), float(row["RotationError"])))
    return out


def frame_data_summary(frame_csv: str) -> tuple:
    """Return (total localization seconds, mean particles per update) from a frame_data.csv."""
    ts, ns = [], []
    try:
        with open(frame_csv) as f:
            for row in csv.DictReader(f):
                try:
                    ts.append(float(row["TimePerFrame"]))
                except (KeyError, ValueError):
                    pass
                # setup rows count toward the time but carry a sample count, not an update count
                if row.get("FrameID") == "setup":
                    continue
                try:
                    ns.append(int(row["ParticleNumber"]))
                except (KeyError, ValueError):
                    pass
    except FileNotFoundError:
        return None, None
    total = sum(ts) if ts else None
    mean_particles = sum(ns) / len(ns) if ns else None
    return total, mean_particles


def format_pose_metrics(
    errors,
    title: str,
    time_per_frame: float | None = None,
    avg_particles: float | None = None,
) -> str:
    m = compute(errors)
    if m is None:
        return f"=== {title} ===\n  no sequences evaluated."
    lines = [
        f"=== {title} ({m['n']} sequences) ===",
        f"  median translation error : {m['median_pos']:.3f} m",
        f"  median rotation error    : {m['median_rot']:.3f} deg",
        f"  recall @ 25 cm           : {m['recall_pos']:.1f}%",
        f"  recall @ 2 deg           : {m['recall_rot']:.1f}%",
        f"  recall @ (25 cm & 2 deg) : {m['recall_joint']:.1f}%",
    ]
    if time_per_frame is not None:
        lines.append(f"  time per query frame     : {time_per_frame:.3f} s")
    if avg_particles is not None:
        lines.append(f"  avg particles per update : {avg_particles:.0f}")
    return "\n".join(lines)


def print_pose_metrics(
    errors,
    title: str,
    time_per_frame: float | None = None,
    save_path: str | None = None,
    avg_particles: float | None = None,
) -> None:
    block = format_pose_metrics(errors, title, time_per_frame, avg_particles)
    print("\n" + block)
    if save_path:
        with open(save_path, "w") as f:
            f.write(block + "\n")
