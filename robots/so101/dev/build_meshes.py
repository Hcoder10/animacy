#!/usr/bin/env python
"""Decimate the SO-ARM100 SO-101 STLs into robots/so101/meshes (< 6 MB) + ATTRIBUTION.md.

    python robots/so101/dev/build_meshes.py --src <dir with the 13 vendor STLs> [--percent 0.3]

Source files: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101/assets
(Apache-2.0). Needs trimesh + fast-simplification.
"""
from __future__ import annotations

import argparse
import os
import sys

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(os.path.dirname(HERE), "meshes")
BUDGET = 6 * 1024 * 1024
FILES = [
    "base_motor_holder_so101_v1", "base_so101_v2", "motor_holder_so101_base_v1", "motor_holder_so101_wrist_v1",
    "moving_jaw_so101_v1", "rotation_pitch_so101_v1", "sts3215_03a_no_horn_v1", "sts3215_03a_v1",
    "under_arm_so101_v1", "upper_arm_so101_v1", "waveshare_mounting_plate_so101_v2",
    "wrist_roll_follower_so101_v1", "wrist_roll_pitch_so101_v2",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--keep", type=float, default=0.15, help="fraction of triangles to keep (trimesh's `percent` is the fraction REMOVED, so use face_count)")
    a = ap.parse_args()
    os.makedirs(MESH_DIR, exist_ok=True)
    total = 0
    for n in FILES:
        src = os.path.join(a.src, n + ".stl")
        m = trimesh.load(src, force="mesh")
        nf = len(m.faces)
        if nf > 500:
            m = m.simplify_quadric_decimation(face_count=max(400, int(nf * a.keep)))
        dst = os.path.join(MESH_DIR, n + ".stl")
        m.export(dst, file_type="stl")
        total += os.path.getsize(dst)
        print(f"  {n:38s} {nf:7d} -> {len(m.faces):6d} faces  {os.path.getsize(dst) / 1024:7.1f} KB")
    print(f"{len(FILES)} meshes, {total / 1024 / 1024:.2f} MB (budget 6 MB)")
    with open(os.path.join(MESH_DIR, "ATTRIBUTION.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "# Mesh attribution\n\n"
            "STL files from **TheRobotStudio / SO-ARM100** (`Simulation/SO101/assets/`), licensed\n"
            "**Apache-2.0** (repository LICENSE, fetched 2026-08-26).\n\n"
            "Source: https://github.com/TheRobotStudio/SO-ARM100\n\n"
            f"Decimated to ~{int(a.keep * 100)}% of their triangle count by `robots/so101/dev/build_meshes.py`\n"
            "(trimesh + fast-simplification) to keep the folder under 6 MB for the browser viewer.\n"
            "`urdf/so101.urdf` is the repo's `so101_new_calib.urdf` with mesh paths pointed here.\n"
        )
    if total > BUDGET:
        print("ERROR: over budget", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
