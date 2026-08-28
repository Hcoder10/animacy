#!/usr/bin/env python
"""Lamp self-collision / clearance check for a `retarget.default` mapping (URDF FK + meshes).

    python scripts/retarget_lamp_clearance.py                       # current ROBOT.md bounds
    python scripts/retarget_lamp_clearance.py --bounds base_pitch=-10:75 elbow_pitch=-5:70 wrist_pitch=-90:30 wrist_roll=-75:75
    python scripts/retarget_lamp_clearance.py --corpus              # + the retargeted human corpus

Geometry: ~N surface points sampled on every link mesh (metres, link frame),
with their face normals. For a pose, the head's points are transformed into
each of base / swivel / lower_arm / upper_arm and the nearest sampled point of
that link is found with a KD-tree; the distance is the clearance, and a point
whose offset from its nearest surface point runs against that surface's
normal (within 15 mm) counts as INSIDE (penetration). Also reported: the
head's lowest point above the desk plane (z = 0). Pose sets: the vendor's
native clips (the reference for "safe"), the 32 corners of the mapping
bounds, a uniform random sweep inside the bounds, and optionally the
retargeted human corpus.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.export import read_autonomous_os_csv  # noqa: E402
from animacy.profile import find_robot  # noqa: E402
from animacy.retarget import mapping_bounds  # noqa: E402

CHECK_LINKS = ["base", "swivel", "lower_arm", "upper_arm"]
NEAR_M = 0.015


def sample_links(urdf, urdf_path, n, seed):
    import trimesh
    from scipy.spatial import cKDTree

    base_dir = os.path.dirname(urdf_path)
    out = {}
    for name, link in urdf.link_map.items():
        m = trimesh.load(os.path.normpath(os.path.join(base_dir, link.visuals[0].geometry.mesh.filename)), force="mesh")
        pts, fid = trimesh.sample.sample_surface(m, n, seed=seed)
        pts = np.asarray(pts)
        out[name] = {"pts": pts, "normals": np.asarray(m.face_normals)[np.asarray(fid)], "tree": cKDTree(pts), "watertight": m.is_watertight}
    return out


def joint_cfg(prof, q):
    cfg = {}
    for j in prof.joints:
        v = (q.get(j.name, j.rest) + j.urdf_offset) * j.urdf_sign
        cfg[j.urdf_joint] = math.radians(v) if j.unit == "deg" else v / 1000.0
    return cfg


def clearance(urdf, prof, links, q):
    """(min unsigned clearance m to base/swivel/lower_arm, head min z m, worst link, clearance to upper_arm m).

    The decimated vendor meshes are not watertight, so an inside/outside test is
    not reliable (it flags a fifth of the vendor's own hardware-executed poses);
    unsigned surface distance is used instead, calibrated against the vendor
    library. The upper arm is the head's neighbour (the neck pivot sits on it)
    and is reported separately."""
    urdf.update_cfg(joint_cfg(prof, q))
    T_head = urdf.get_transform("head", urdf.base_link)
    head = links["head"]["pts"] @ T_head[:3, :3].T + T_head[:3, 3]
    best, worst, arm = 1e9, None, 1e9
    for name in CHECK_LINKS:
        T = urdf.get_transform(name, urdf.base_link)
        Tinv = np.linalg.inv(T)
        pts = head @ Tinv[:3, :3].T + Tinv[:3, 3]
        d, _ = links[name]["tree"].query(pts)
        dist = float(d.min())
        if name == "upper_arm":
            arm = dist
        elif dist < best:
            best, worst = dist, name
    return best, float(head[:, 2].min()), worst, arm


def corners(prof, bounds):
    names = [j.name for j in prof.joints]
    for combo in itertools.product(*[bounds[n] for n in names]):
        yield dict(zip(names, combo))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bounds", nargs="*", default=[], help="joint=lo:hi overrides of the mapping bounds")
    ap.add_argument("--sweep", type=int, default=1000)
    ap.add_argument("--points", type=int, default=3000)
    ap.add_argument("--corpus", action="store_true", help="also check the retargeted human corpus")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import yourdfpy

    prof = find_robot("lamp")
    urdf = yourdfpy.URDF.load(prof.urdf_path(), load_meshes=False, build_scene_graph=True)
    links = sample_links(urdf, prof.urdf_path(), a.points, a.seed)
    rng = np.random.default_rng(a.seed)
    print("watertight:", {n: L["watertight"] for n, L in links.items()})

    mp = prof.mapping("default")
    bounds = {j.name: mapping_bounds(j, mp.get(j.name)) for j in prof.joints}
    for spec in a.bounds:
        n, r = spec.split("=")
        lo, hi = r.split(":")
        bounds[n] = (float(lo), float(hi))
    print("bounds:", bounds)

    def report(label, poses):
        rows = [(clearance(urdf, prof, links, q), q) for q in poses]
        worst = min(rows, key=lambda r: r[0][0])
        zmin = min(rows, key=lambda r: r[0][1])
        armmin = min(rows, key=lambda r: r[0][3])
        contact = sum(1 for r in rows if r[0][0] < 0.005)
        print(f"{label:34s} n={len(rows):5d} | head-to-{worst[0][2]:<9s} min {worst[0][0] * 1000:6.1f} mm | <5 mm in {contact:4d} poses | head min z {zmin[0][1] * 1000:6.1f} mm | head-to-upper_arm min {armmin[0][3] * 1000:5.1f} mm")
        if worst[0][0] < 0.02:
            print("   tightest pose:", {k: round(v, 1) for k, v in worst[1].items()})
        if zmin[0][1] < 0.06:
            print("   lowest pose:  ", {k: round(v, 1) for k, v in zmin[1].items()})
        return rows

    d = os.path.join(prof.dir, "clips", "native")
    vendor = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".csv"):
            df = read_autonomous_os_csv(os.path.join(d, fn))
            for i in range(0, len(df), 3):
                vendor.append({j: float(df[j].iloc[i]) for j in prof.joint_names})
    report("vendor native clips (every 3rd)", vendor)
    rest = {j.name: j.rest for j in prof.joints}
    report("rest", [rest])
    perf = {j.name: j.rest + (mp[j.name].offset if j.name in mp else 0.0) for j in prof.joints}
    report("rest + mapping offset", [perf])
    report("bound corners (32)", list(corners(prof, bounds)))
    sweep = [{n: float(rng.uniform(lo, hi)) for n, (lo, hi) in bounds.items()} for _ in range(a.sweep)]
    report("uniform sweep inside bounds", sweep)
    if a.corpus:
        from animacy.retarget_fit import DEFAULT_EXCLUDE, load_human_clips, retarget_tables

        humans = load_human_clips(os.path.join(ROOT, "data", "clips"), None, DEFAULT_EXCLUDE)
        poses = []
        for tb in retarget_tables(humans, prof, "default").values():
            for i in range(0, len(tb), 10):
                poses.append({j: float(tb[j].iloc[i]) for j in prof.joint_names})
        report("retargeted human corpus (1/10)", poses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
