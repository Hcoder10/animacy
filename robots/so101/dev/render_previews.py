#!/usr/bin/env python
"""Render SO-101 preview PNGs with matplotlib (no GL): rest, look-left, puppet wave.

Poses go through the profile exactly like the pipeline: canonical channels ->
``animacy.retarget.raw_joint_targets`` (mapping arithmetic) -> ``to_urdf_values``
(sign/offset/units) -> yourdfpy FK. So the PNGs check the ROBOT.md mapping, not
just the URDF.

    python robots/so101/dev/render_previews.py [--out robots/so101/urdf/preview]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RDIR))
sys.path.insert(0, ROOT)

from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import raw_joint_targets, to_urdf_values  # noqa: E402
from animacy.schema import CHANNELS  # noqa: E402

FACE_BUDGET = 900
GAZE_LINK = "gripper_frame_link"   # its +z is the gripper's pointing direction

# (name, mode, canonical channel values) — everything not given is 0 (= rest through the mapping)
SCENES = [
    ("00_rest", "default", {}),
    ("01_look_left", "default", {"head_yaw": 60.0}),
    ("02_look_up", "default", {"head_pitch": 40.0}),
    ("03_lean_in_talking", "default", {"torso_lean_fwd": 30.0, "head_x": 60.0, "mouth_open": 0.8, "brow_l": 1.0, "brow_r": 1.0}),
    # a wave: upper arm ~horizontal (pitch 100), elbow bent so the forearm stands up, hand open
    ("04_puppet_wave", "puppet", {"shoulder_yaw": 15.0, "shoulder_pitch": 100.0, "elbow_flex": 80.0, "wrist_pitch": 10.0, "hand_open": 0.9}),
]


def load_robot(profile):
    import trimesh
    import yourdfpy

    urdf = profile.urdf_path()
    u = yourdfpy.URDF.load(urdf, load_meshes=True, build_scene_graph=True,
                           filename_handler=lambda fname: yourdfpy.filename_handler_relative(fname, dir=os.path.dirname(urdf)))
    for name, g in list(u.scene.geometry.items()):
        if isinstance(g, trimesh.Trimesh) and len(g.faces) > FACE_BUDGET:
            try:
                u.scene.geometry[name] = g.simplify_quadric_decimation(face_count=FACE_BUDGET)
            except Exception:
                pass
    return u


def joints_for(profile, mode, channels):
    row = {c: 0.0 for c in CHANNELS}
    row.update(channels)
    row["t"] = 0.0
    frames = pd.DataFrame({k: [v] for k, v in row.items()})
    table = raw_joint_targets(frames, profile, mode)
    return {j: float(table[j].iloc[0]) for j in profile.joint_names}, table


def pose(u, profile, table):
    vals = to_urdf_values(table, profile)
    u.update_cfg({k: float(v[0]) for k, v in vals.items()})


def link_of_node(u, node):
    g = u.scene.graph
    n = node
    while n is not None and n not in u.link_map:
        n = g.transforms.parents.get(n)
    return n or node


def triangles(u):
    import trimesh

    palette = {"base_link": (0.35, 0.35, 0.38), "shoulder_link": (0.55, 0.55, 0.6), "upper_arm_link": (0.85, 0.55, 0.2),
               "lower_arm_link": (0.9, 0.7, 0.3), "wrist_link": (0.6, 0.6, 0.65), "gripper_link": (0.2, 0.45, 0.85),
               "moving_jaw_so101_v1_link": (0.85, 0.2, 0.2)}
    out = []
    g = u.scene.graph
    for node in g.nodes_geometry:
        T, gname = g[node]
        geom = u.scene.geometry[gname]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        v = trimesh.transform_points(geom.vertices, T)
        out.append((v[geom.faces], palette.get(link_of_node(u, node), (0.6, 0.6, 0.6))))
    return out


def draw(ax, tris, u, view, title):
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ls = LightSource(azdeg=315, altdeg=45)
    for faces, col in tris:
        n = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        shade = ls.shade_normals(n, fraction=1.0)
        cols = np.clip(np.asarray(col)[None, :] * (0.45 + 0.55 * shade[:, None]), 0, 1)
        ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="none"))
    L = 0.1
    for d, c in (((L, 0, 0), "r"), ((0, L, 0), "g"), ((0, 0, L), "b")):
        ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=c, lw=2.5)
    T = u.get_transform(GAZE_LINK, u.base_link)
    o, z = T[:3, 3], T[:3, 2]
    ax.plot([o[0], o[0] + 0.15 * z[0]], [o[1], o[1] + 0.15 * z[1]], [o[2], o[2] + 0.15 * z[2]], color="m", lw=3)
    ax.set_xlim(-0.15, 0.35)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(0.0, 0.6)
    ax.set_box_aspect((0.5, 0.5, 0.6))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x fwd")
    ax.set_ylabel("y left")
    ax.set_zlabel("z up")
    ax.set_title(title, fontsize=9)


def render(u, profile, name, mode, channels, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joints, table = joints_for(profile, mode, channels)
    pose(u, profile, table)
    tris = triangles(u)
    fig = plt.figure(figsize=(15, 5.2))
    for i, (t, v) in enumerate((("iso from front-left", (25, 35)), ("front (camera on +x)", (10, 0)), ("robot's left side (camera on +y)", (10, 90)))):
        draw(fig.add_subplot(1, 3, i + 1, projection="3d"), tris, u, v, t)
    fig.suptitle(f"{name} [{mode}]  " + ", ".join(f"{k}={v:.0f}" for k, v in joints.items()), fontsize=10)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.png")
    fig.savefig(p, dpi=80)
    plt.close(fig)
    T = u.get_transform(GAZE_LINK, u.base_link)
    print(f"wrote {p}  joints {joints}  tip {np.round(T[:3, 3], 3).tolist()} gaze {np.round(T[:3, 2], 2).tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RDIR, "urdf", "preview"))
    ap.add_argument("--only")
    a = ap.parse_args()
    profile = load_profile(RDIR)
    u = load_robot(profile)
    for name, mode, ch in SCENES:
        if a.only and name not in a.only.split(","):
            continue
        render(u, profile, name, mode, ch, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
