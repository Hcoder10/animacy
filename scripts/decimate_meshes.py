#!/usr/bin/env python
"""Shrink a folder of meshes to a byte budget for the browser viewer.

    python scripts/decimate_meshes.py <in_dir> <out_dir> --budget-mb 6 [--scale 0.001] [--keep 0.3] [--min-faces 300]

Every ``.stl``/``.obj``/``.ply`` in ``in_dir`` is written to ``out_dir`` as a
binary STL (same file stem, ``.stl``), decimated so the folder fits the
budget. A binary STL is 84 + 50*faces bytes, so the budget is a *face* budget
and the keep fraction is computed directly (no trial and error); ``--keep``
overrides it. ``--scale`` multiplies vertices first (``0.001`` for meshes
modelled in millimetres when the URDF is in metres).

GOTCHA (documented because it cost a retry on every robot so far): trimesh's
``simplify_quadric_decimation(percent=p)`` — ``percent`` is the fraction of
faces to REMOVE, not to keep (``percent=0.3`` leaves 70 %). This script always
passes ``face_count`` instead. Needs ``trimesh`` + ``fast-simplification``.

Attribution is the author's job: copy the vendor's license into
``<out_dir>/ATTRIBUTION.md`` (see robots/*/meshes/ATTRIBUTION.md for the shape).
"""
from __future__ import annotations

import argparse
import os
import sys

EXTS = (".stl", ".obj", ".ply")
STL_HEADER = 84
STL_PER_FACE = 50


def decimate_dir(in_dir: str, out_dir: str, budget_mb: float = 6.0, scale: float = 1.0, keep: float | None = None,
                 min_faces: int = 300, quiet: bool = False) -> dict:
    import trimesh

    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit(f"no {EXTS} files in {in_dir}")
    meshes = {}
    for f in files:
        m = trimesh.load(os.path.join(in_dir, f), force="mesh")
        if scale != 1.0:
            m.apply_scale(scale)
        meshes[f] = m
    total_faces = sum(len(m.faces) for m in meshes.values())
    budget = int(budget_mb * 1024 * 1024)
    face_budget = (budget - STL_HEADER * len(files)) // STL_PER_FACE
    k = keep if keep is not None else min(1.0, face_budget / max(total_faces, 1))
    os.makedirs(out_dir, exist_ok=True)
    report = {"files": [], "keep": k, "budget_bytes": budget}
    for attempt in range(6):
        total = 0
        report["files"] = []
        for f, m in meshes.items():
            nf = len(m.faces)
            target = nf
            out_m = m
            if nf > min_faces and k < 1.0:
                target = max(min_faces, int(nf * k))
                if target < nf:
                    out_m = m.simplify_quadric_decimation(face_count=target)
            dst = os.path.join(out_dir, os.path.splitext(f)[0] + ".stl")
            out_m.export(dst, file_type="stl")
            size = os.path.getsize(dst)
            total += size
            report["files"].append({"file": os.path.basename(dst), "faces_in": nf, "faces_out": len(out_m.faces), "bytes": size})
        report["total_bytes"] = total
        report["keep"] = k
        if total <= budget or keep is not None:
            break
        k *= 0.9 * budget / total  # min_faces floors pushed us over; tighten and retry
    if not quiet:
        for r in report["files"]:
            print(f"  {r['file']:40s} {r['faces_in']:8d} -> {r['faces_out']:7d} faces  {r['bytes'] / 1024:8.1f} KB")
        print(f"{len(files)} meshes, {total / 1024 / 1024:.2f} MB (budget {budget_mb:g} MB, keep {k:.3f})")
    if report["total_bytes"] > budget:
        print("WARNING: over budget even at the min-faces floor; lower --min-faces", file=sys.stderr)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--budget-mb", type=float, default=6.0)
    ap.add_argument("--scale", type=float, default=1.0, help="vertex scale applied first (0.001 = mm -> m)")
    ap.add_argument("--keep", type=float, default=None, help="fixed fraction of faces to keep (overrides the budget)")
    ap.add_argument("--min-faces", type=int, default=300)
    a = ap.parse_args(argv)
    rep = decimate_dir(a.in_dir, a.out_dir, a.budget_mb, a.scale, a.keep, a.min_faces)
    return 0 if rep["total_bytes"] <= rep["budget_bytes"] else 1


if __name__ == "__main__":
    sys.exit(main())
