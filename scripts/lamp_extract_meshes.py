"""Extract per-link visual meshes for the Autonomous Lamp URDF from the vendor's
``lamp.glb`` (Autonomous OS, Apache-2.0).

The GLB is a Blender export of the vendor CAD assembly with one skinned mesh
group per moving part (``0_base``, ``1_base_yaw`` ... ``5_wrist_pitch``) and an
armature whose bones sit on the joint pivots. Each group is exported here as
one binary STL in *metres*, expressed in the frame of the link that carries it:
origin at that link's joint pivot (from the armature), axes = the URDF world
axes at the CAD ("bind") pose. So a mesh's ``<origin>`` in the URDF is
identity and all pose information lives in the joints.

GLB frame is glTF (y up, lamp faces +x). URDF frame is x forward, y left, z up:
    urdf = (x_glb, -z_glb, y_glb)

Run:  python scripts/lamp_extract_meshes.py   (writes robots/lamp/meshes/*.stl)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GLB = os.path.join(ROOT, "robots", "lamp", "cad_src", "lamp.glb")
OUT = os.path.join(ROOT, "robots", "lamp", "meshes")

# GLB -> URDF axis change (x, y, z)_glb -> (x, -z, y)_urdf
GLB2URDF = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)

# mesh group prefix -> (stl name, armature bone whose head is the link origin)
# The wire (6_wire) spans every link in the bind pose and is not exported.
GROUPS = {
    "0_base": ("base", None),
    "1_base_yaw": ("swivel", "base_yaw"),
    "2_base_pitch": ("lower_arm", "base_pitch"),
    "3_elbow_pitch": ("upper_arm", "elbow_pitch"),
    "4_wrist_roll": ("neck", "wrist_pitch"),   # bone at the neck base (pitch axle); drives the neck group
    "5_wrist_pitch": ("head", "wrist_roll"),   # bone at the head (roll about the neck); drives the head group
}
# face budget per link after decimation (binary STL = 50 B/face). Decimation is
# done per solid (never below MIN_KEEP of a solid's faces) because decimating the
# concatenated, non-watertight soup collapsed thin shells into garbage.
BUDGET = {"base": 20000, "swivel": 12000, "lower_arm": 6600, "upper_arm": 12000, "neck": 6000, "head": 20000}
MIN_KEEP = 0.35
# The base group is 69 solids, most of them electronics hidden inside the drum.
# Keep only the shell, the top plate and the feet.
BASE_KEEP = {"Solid2.206", "Solid2.206_61", "Solid2.206_29"}


def pivots_urdf(scene: trimesh.Scene) -> dict:
    """Bone head positions in URDF frame (metres), from the armature node transforms."""
    out = {}
    for b in ["base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]:
        T = scene.graph.get(b)[0]
        out[b] = (GLB2URDF @ np.append(T[:3, 3], 1.0))[:3]
    # the yaw axis is the lamp's vertical centreline; snap the tiny CAD offsets
    out["base_yaw"][0] = 0.0
    out["base_pitch"][0] = 0.0
    for b in out:
        out[b][1] = 0.0
    return out


def main() -> int:
    scene = trimesh.load(GLB)
    piv = pivots_urdf(scene)
    os.makedirs(OUT, exist_ok=True)
    groups = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        key = next((k for k in GROUPS if node.startswith(k)), None)
        if key is None:
            continue
        if key == "0_base" and gname not in BASE_KEEP:
            continue
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        groups.setdefault(key, []).append(g)
    total = 0
    for key, (name, bone) in GROUPS.items():
        solids = groups[key]
        before = sum(len(g.faces) for g in solids)
        if before > BUDGET[name]:
            ratio = BUDGET[name] / before
            out = []
            for g in solids:
                target = int(len(g.faces) * max(ratio, MIN_KEEP))
                if len(g.faces) > 400 and target < len(g.faces):
                    g = g.simplify_quadric_decimation(face_count=target)
                out.append(g)
            solids = out
        m = trimesh.util.concatenate(solids)
        m.apply_transform(GLB2URDF)
        origin = piv[bone] if bone else np.zeros(3)
        m.apply_translation(-origin)
        m.remove_unreferenced_vertices()
        path = os.path.join(OUT, f"{name}.stl")
        m.export(path)
        size = os.path.getsize(path)
        total += size
        print(f"{name:10s} faces {before:7d} -> {len(m.faces):6d}  {size/1024:7.1f} KB  origin(urdf)={np.round(origin, 4)}  bounds={np.round(m.bounds, 3).tolist()}")
    print(f"total {total/1024/1024:.2f} MB")
    print("pivots (urdf frame):", {k: np.round(v, 4).tolist() for k, v in piv.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
