"""
Extract color, depth, poses and intrinsics from a ScanNet .sens file.

Adapted from the ScanNet SensReader:
https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py

Writes color/<frame>.jpg, depth/<frame>.png (16 bit), pose/<frame>.txt and
intrinsic/{intrinsic,extrinsic}_{color,depth}.txt into the output directory.

Usage:
    python -m sg2loc.scannet.preprocessing.extract_sens <scan.sens> <output-dir>
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import struct
import zlib

import numpy as np
import png
from imageio.v2 import imread, imwrite

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    """One frame record of a .sens file: pose, timestamps and compressed images."""

    def load(self, f) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", f.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", f.read(8))[0]
        color_size_bytes = struct.unpack("Q", f.read(8))[0]
        depth_size_bytes = struct.unpack("Q", f.read(8))[0]
        self.color_data = f.read(color_size_bytes)
        self.depth_data = f.read(depth_size_bytes)


class SensorData:
    """A parsed .sens file (format version 4)."""

    def __init__(self, filename: str, header_only: bool = False):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert version == 4, f"unsupported .sens version {version}"
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("ascii", "replace")
            self.intrinsic_color = self._read_mat4(f)
            self.extrinsic_color = self._read_mat4(f)
            self.intrinsic_depth = self._read_mat4(f)
            self.extrinsic_depth = self._read_mat4(f)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack("i", f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack("i", f.read(4))[0]]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            if header_only:
                return
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    @staticmethod
    def _read_mat4(f) -> np.ndarray:
        return np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)

    def export_depth_images(self, output_path: str) -> None:
        assert self.depth_compression_type == "zlib_ushort", self.depth_compression_type
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames)} depth frames to {output_path}")
        for i, frame in enumerate(self.frames):
            depth = np.frombuffer(zlib.decompress(frame.depth_data), dtype=np.uint16)
            depth = depth.reshape(self.depth_height, self.depth_width)
            with open(osp.join(output_path, f"{i}.png"), "wb") as f:
                writer = png.Writer(width=self.depth_width, height=self.depth_height, bitdepth=16)
                writer.write(f, depth.tolist())

    def export_color_images(self, output_path: str) -> None:
        assert self.color_compression_type == "jpeg", self.color_compression_type
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames)} color frames to {output_path}")
        for i, frame in enumerate(self.frames):
            imwrite(osp.join(output_path, f"{i}.jpg"), imread(frame.color_data))

    @staticmethod
    def _save_mat_to_file(matrix: np.ndarray, filename: str) -> None:
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting {len(self.frames)} camera poses to {output_path}")
        for i, frame in enumerate(self.frames):
            self._save_mat_to_file(frame.camera_to_world, osp.join(output_path, f"{i}.txt"))

    def export_intrinsics(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        print(f"exporting camera intrinsics to {output_path}")
        self._save_mat_to_file(self.intrinsic_color, osp.join(output_path, "intrinsic_color.txt"))
        self._save_mat_to_file(self.extrinsic_color, osp.join(output_path, "extrinsic_color.txt"))
        self._save_mat_to_file(self.intrinsic_depth, osp.join(output_path, "intrinsic_depth.txt"))
        self._save_mat_to_file(self.extrinsic_depth, osp.join(output_path, "extrinsic_depth.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sens_file", help="the .sens file of a scan")
    parser.add_argument("output_dir", help="scan directory that receives the extracted frames")
    parser.add_argument(
        "--intrinsics-only",
        action="store_true",
        help="only write the intrinsic files, the input may be a truncated .sens header",
    )
    args = parser.parse_args()
    print(f"loading {args.sens_file}")
    if args.intrinsics_only:
        SensorData(args.sens_file, header_only=True).export_intrinsics(
            osp.join(args.output_dir, "intrinsic")
        )
        return
    data = SensorData(args.sens_file)
    data.export_depth_images(osp.join(args.output_dir, "depth"))
    data.export_color_images(osp.join(args.output_dir, "color"))
    data.export_poses(osp.join(args.output_dir, "pose"))
    data.export_intrinsics(osp.join(args.output_dir, "intrinsic"))


if __name__ == "__main__":
    main()
