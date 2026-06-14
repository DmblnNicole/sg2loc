"""
Generate SceneGraphFusion predictions for the ScanNet evaluation scans.

Drives the externally built SceneGraphFusion executable (clone and build
https://github.com/ShunChengWu/SceneGraphFusion with USE_RENDEREDVIEW and assimp, and
download its traced model). Writes scene_graph_fusion/predictions.json, inseg.ply and
node_semantic.ply per scan.
"""

import argparse
import os.path as osp
import subprocess


def run_batch(commands: list, jobs: int) -> None:
    for start in range(0, len(commands), jobs):
        procs = [subprocess.Popen(c, shell=True) for c in commands[start : start + jobs]]
        for p in procs:
            p.wait()
        print(
            f"[sgfusion] {min(start + jobs, len(commands))}/{len(commands)} launched batches done"
        )


def outputs_complete(out_folder: str) -> bool:
    files = ("predictions.json", "inseg.ply", "node_semantic.ply")
    return all(osp.isfile(osp.join(out_folder, f)) for f in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans-dir", required=True, help="directory with <scene>/<scene>.sens")
    parser.add_argument("--scene-list", required=True, help="txt file, one scan id per line")
    parser.add_argument("--exe", required=True, help="built SceneGraphFusion exe_GraphSLAM")
    parser.add_argument("--model", required=True, help="SceneGraphFusion traced model directory")
    parser.add_argument("--jobs", type=int, default=4, help="parallel SceneGraphFusion processes")
    parser.add_argument("--retries", type=int, default=2, help="re-attempts for failed scans")
    args = parser.parse_args()

    scan_ids = [ln.strip() for ln in open(args.scene_list) if ln.strip()]
    # the list holds the _00 query scans, each room's _01 map twin is processed too
    scan_ids = [s for q in scan_ids for s in (q, q[:-3] + "_01")]
    todo, skipped_done, skipped_no_sens = [], 0, 0
    for scan_id in scan_ids:
        out_folder = osp.join(args.scans_dir, scan_id, "scene_graph_fusion")
        if outputs_complete(out_folder):
            skipped_done += 1
            continue
        if not osp.isfile(osp.join(args.scans_dir, scan_id, f"{scan_id}.sens")):
            skipped_no_sens += 1
            continue
        todo.append(scan_id)
    print(
        f"[sgfusion] to generate: {len(todo)} (done: {skipped_done}, no .sens: {skipped_no_sens})"
    )

    for attempt in range(1 + args.retries):
        commands = []
        for scan_id in todo:
            out_folder = osp.join(args.scans_dir, scan_id, "scene_graph_fusion")
            if outputs_complete(out_folder):
                continue
            sens = osp.join(args.scans_dir, scan_id, f"{scan_id}.sens")
            commands.append(
                f"{args.exe} --pth_in {sens} --pth_out {out_folder} --pth_model {args.model}"
            )
        if not commands:
            break
        print(f"[sgfusion] attempt {attempt + 1}: {len(commands)} scans")
        run_batch(commands, args.jobs)

    failed = [
        s for s in todo if not outputs_complete(osp.join(args.scans_dir, s, "scene_graph_fusion"))
    ]
    print(f"[sgfusion] complete, failed scans: {failed if failed else 'none'}")


if __name__ == "__main__":
    main()
