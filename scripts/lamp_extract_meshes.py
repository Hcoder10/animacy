"""Build per-link visual meshes for the Autonomous Lamp URDF from the vendor's
per-part STLs (Autonomous OS, Apache-2.0), placed with the vendor's ``lamp.glb``.

Why two sources: the per-part STLs (``cad_src/*.stl``, millimetres) are clean,
watertight printed-part shells, but each is exported in its own local frame so
nothing says where it sits in the assembled lamp. ``lamp.glb`` is the assembled
lamp (glTF, metres, y up) with one skinned mesh group per moving part and an
armature whose bones sit on the joint pivots -- but its solids are fragmented
open surfaces that render badly. So: the GLB provides the placement (pivots +
a registration target per part), the STLs provide the surfaces.

Placement of every STL unit is recovered by rigid registration (PCA-initialised
ICP, 4 proper sign combinations, best residual kept) of the STL onto the
matching GLB solid of the same link group. Residuals are printed; anything
over ~1.5 mm should be looked at.

Frames: GLB (x, y, z) -> URDF (x, -z, y) [x forward, y left, z up]. Each link
mesh is expressed in its link frame: origin at the link's joint pivot, axes =
URDF world axes at the CAD ("bind") pose, so mesh origins in the URDF are
identity and all pose information lives in the joints.

Run:  python scripts/lamp_extract_meshes.py     (writes robots/lamp/meshes/*.stl)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import trimesh
import trimesh.registration as reg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CAD = os.path.join(ROOT, "robots", "lamp", "cad_src")
GLB = os.path.join(CAD, "lamp.glb")
OUT = os.path.join(ROOT, "robots", "lamp", "meshes")

GLB2URDF = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
MM = 0.001

# link -> (GLB group prefix, armature bone at the link pivot)
LINKS = {
    "base": ("0_base", None),
    "swivel": ("1_base_yaw", "base_yaw"),
    "lower_arm": ("2_base_pitch", "base_pitch"),
    "upper_arm": ("3_elbow_pitch", "elbow_pitch"),
    "neck": ("4_wrist_roll", "wrist_pitch"),   # neck group is driven by the neck-base (pitch) bone
    "head": ("5_wrist_pitch", "wrist_roll"),   # head group is driven by the head (roll) bone
}
# STL "units": parts that share one local frame and are registered together.
# (register: the parts whose union is matched against a GLB solid; extra: parts
# in the same frame that ride along with the recovered transform)
UNITS = [
    # link, register parts, extra parts
    ("base", ["base"], ["base-cap", "button"]),
    ("swivel", ["swivel-part-part1", "swivel-part-part2", "swivel-part-part3"], []),
    ("lower_arm", ["arm-1-part1", "arm-1-part2"], []),
    ("upper_arm", ["arm-2-part1", "arm-2-part2"], []),
    ("neck", ["neck"], []),
    ("neck", ["cap-servo"], []),
    ("head", ["head-part2", "head-part3"], ["head-part1", "light-cover"]),
]
# faces kept per part after decimation (only parts above this are decimated)
MAX_FACES = 9000
SAMPLES = 4000


def load_glb_groups():
    scene = trimesh.load(GLB)
    piv = {}
    for b in ["base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]:
        T = scene.graph.get(b)[0]
        p = (GLB2URDF @ np.append(T[:3, 3], 1.0))[:3]
        p[1] = 0.0                       # the CAD arm plane is exactly y = 0
        piv[b] = p
    piv["base_yaw"][0] = 0.0
    piv["base_pitch"][0] = 0.0           # yaw axis = lamp centreline
    groups = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        key = next((k for k in [v[0] for v in LINKS.values()] if node.startswith(k)), None)
        if key is None:
            continue
        g = scene.geometry[gname].copy()
        g.apply_transform(GLB2URDF @ T)
        groups.setdefault(key, []).append((gname, g))
    return piv, groups


def _pca_frame(pts):
    c = pts.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((pts - c).T))
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    return c, V


def register(src: trimesh.Trimesh, tgt: trimesh.Trimesh):
    """Rigid transform (4x4) taking ``src`` onto ``tgt`` and the mean residual (m)."""
    sp = src.sample(SAMPLES)
    cs, Vs = _pca_frame(sp)
    ct, Vt = _pca_frame(tgt.sample(SAMPLES))
    best = None
    for signs in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        R0 = Vt @ (Vs * np.array(signs)).T
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = ct - R0 @ cs
        M, _, cost = reg.icp(sp, tgt, initial=T0, max_iterations=60, threshold=1e-7, scale=False)
        if best is None or cost < best[1]:
            best = (M, cost)
    return best


def main() -> int:
    piv, groups = load_glb_groups()
    os.makedirs(OUT, exist_ok=True)
    link_parts = {k: [] for k in LINKS}
    for link, reg_parts, extra in UNITS:
        parts = {n: trimesh.load(os.path.join(CAD, f"{n}.stl"), force="mesh") for n in reg_parts + extra}
        for m in parts.values():
            m.apply_scale(MM)
        src = trimesh.util.concatenate([parts[n] for n in reg_parts])
        ext = np.sort(src.extents)
        # candidate GLB solids: same group, comparable size
        best = None
        for gname, g in groups[LINKS[link][0]]:
            ge = np.sort(g.extents)
            if len(g.faces) < 300 or np.any(ge < 0.5 * ext) or np.any(ge > 1.6 * ext):
                continue
            M, cost = register(src, g)
            if best is None or cost < best[1]:
                best = (M, cost, gname)
        if best is None:
            raise SystemExit(f"{link}: no GLB solid matches {reg_parts} (extents {np.round(ext*1000,1)} mm)")
        M, cost, gname = best
        print(f"{link:10s} {'+'.join(reg_parts):40s} -> {gname:14s} residual {cost*1000:5.2f} mm")
        for n, m in parts.items():
            m.apply_transform(M)
            link_parts[link].append((n, m))
    total = 0
    for link, (group, bone) in LINKS.items():
        origin = piv[bone] if bone else np.zeros(3)
        outm = []
        for n, m in link_parts[link]:
            if len(m.faces) > MAX_FACES:
                m = m.simplify_quadric_decimation(face_count=MAX_FACES)
            m.apply_translation(-origin)
            outm.append(m)
        m = trimesh.util.concatenate(outm)
        m.remove_unreferenced_vertices()
        path = os.path.join(OUT, f"{link}.stl")
        m.export(path)
        size = os.path.getsize(path)
        total += size
        wt = all(p.is_watertight for p in outm)
        print(f"  wrote {link:10s} faces={len(m.faces):6d} {size/1024:7.1f} KB watertight_parts={wt} origin={np.round(origin,4)} bounds={np.round(m.bounds,3).tolist()}")
    print(f"total {total/1024/1024:.2f} MB")
    print("pivots (urdf frame):", {k: np.round(v, 4).tolist() for k, v in piv.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
