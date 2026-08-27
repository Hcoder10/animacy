#!/usr/bin/env python
"""Render keyframes of the Reachy Mini visualization URDF to PNG (matplotlib, no GL).

Poses are given in ROBOT.md units and animacy signs (mm / deg, +yaw = left,
+pitch = UP, antennas + = outward) and pushed through the profile's
``urdf_sign``/``urdf_offset`` exactly like the retarget pipeline
(``animacy.retarget.to_urdf_values``), so what you see is what a joint table
would show in the browser viewer.

    python scripts/reachy_render_clip.py                      # the fixed sign-check poses
    python scripts/reachy_render_clip.py --clip robots/reachy_mini/clips/native/amazed1.json --frames 3

Writes to robots/reachy_mini/urdf/preview/*.png. Each PNG has three views with
the world axes drawn (x red = forward, y green = robot's left, z blue = up) and
a magenta gaze ray along the head's +x.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import to_urdf_values  # noqa: E402

RDIR = os.path.join(ROOT, "robots", "reachy_mini")
PREVIEW = os.path.join(RDIR, "urdf", "preview")
FACE_BUDGET = 700  # triangles per geometry for matplotlib

LINK_COLORS = {
    "base": (0.35, 0.35, 0.38), "body": (0.55, 0.55, 0.58), "stewart_static": (0.25, 0.30, 0.40),
    "head": (0.93, 0.93, 0.93), "antenna_left_link": (0.85, 0.20, 0.20), "antenna_right_link": (0.15, 0.35, 0.85),
}

CHECK_POSES = [
    ("00_rest", {}),
    ("01_head_yaw_p40", {"head_yaw": 40}),
    ("02_head_pitch_p25", {"head_pitch": 25}),
    ("03_head_roll_p20", {"head_roll": 20}),
    ("04_antennas_p90", {"antenna_left": 90, "antenna_right": 90}),
    ("05_body_yaw_p60", {"body_yaw": 60}),
    ("06_head_xyz_p20mm", {"head_x": 20, "head_y": 20, "head_z": 20}),
]


def load_robot(profile):
    import trimesh
    import yourdfpy

    urdf = profile.urdf_path()
    u = yourdfpy.URDF.load(urdf, load_meshes=True, build_scene_graph=True,
                           filename_handler=lambda fname: yourdfpy.filename_handler_relative(fname, dir=os.path.dirname(urdf)))
    # slim the geometries once so matplotlib stays responsive
    for name, g in list(u.scene.geometry.items()):
        if isinstance(g, trimesh.Trimesh) and len(g.faces) > FACE_BUDGET:
            try:
                u.scene.geometry[name] = g.simplify_quadric_decimation(face_count=FACE_BUDGET)
            except Exception:
                pass
    return u


def link_of_node(u, node: str) -> str:
    """Walk up the scene graph until we hit a URDF link name."""
    g = u.scene.graph
    n = node
    while n is not None and n not in u.link_map:
        n = g.transforms.parents.get(n)
    return n or node


def pose_robot(u, profile, joints: dict):
    row = {j.name: float(joints.get(j.name, j.rest)) for j in profile.joints}
    table = pd.DataFrame({"t": [0.0], **{k: [v] for k, v in row.items()}})
    vals = to_urdf_values(table, profile)
    cfg = {k: float(v[0]) for k, v in vals.items()}
    u.update_cfg(cfg)
    return row


def collect_triangles(u):
    import trimesh

    out = []
    g = u.scene.graph
    for node in g.nodes_geometry:
        T, geom_name = g[node]
        geom = u.scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        v = trimesh.transform_points(geom.vertices, T)
        link = link_of_node(u, node)
        col = LINK_COLORS.get(link, (0.6, 0.6, 0.6))
        out.append((v[geom.faces], col))
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
        pc = Poly3DCollection(faces, facecolors=cols, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
    # world axes at the base
    L = 0.09
    for d, c in (((L, 0, 0), "r"), ((0, L, 0), "g"), ((0, 0, L), "b")):
        ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=c, lw=2.5)
    # gaze ray along the head's +x, drawn from the camera (head frame + [0.0395, 0, 0.0525], vendor camera_frame)
    T = u.get_transform("head", u.base_link)
    o = (T @ np.array([0.0395, 0.0, 0.0525, 1.0]))[:3]
    x = T[:3, 0]
    ax.plot([o[0], o[0] + 0.14 * x[0]], [o[1], o[1] + 0.14 * x[1]], [o[2], o[2] + 0.14 * x[2]], color="m", lw=3)
    ax.scatter([T[0, 3]], [T[1, 3]], [T[2, 3]], color="m", s=25)  # SDK head frame origin
    ax.set_xlim(-0.14, 0.14)
    ax.set_ylim(-0.14, 0.14)
    ax.set_zlim(0.0, 0.38)
    ax.set_box_aspect((1, 1, 38 / 28))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x fwd")
    ax.set_ylabel("y left")
    ax.set_zlabel("z up")
    ax.set_title(title, fontsize=9)


def render_pose(u, profile, joints: dict, out_png: str, label: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = pose_robot(u, profile, joints)
    tris = collect_triangles(u)
    fig = plt.figure(figsize=(15, 5.2))
    views = [("iso from front-left", (22, 35)), ("front (camera on +x looking back)", (8, 0)),
             ("robot's left side (camera on +y)", (8, 90))]
    for i, (name, v) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw(ax, tris, u, v, name)
    nz = {k: v for k, v in row.items() if abs(v) > 1e-9}
    fig.suptitle(f"{label}   " + (", ".join(f"{k}={v:g}" for k, v in nz.items()) or "all joints at rest (0)"), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=80)
    plt.close(fig)
    print("wrote", out_png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=RDIR)
    ap.add_argument("--clip", help="animacy json joint table (write_joint_table fmt=json)")
    ap.add_argument("--frames", default="3", help="N evenly spaced frames, or comma-separated times in seconds")
    ap.add_argument("--out", default=PREVIEW)
    ap.add_argument("--only", help="comma-separated subset of the check poses")
    a = ap.parse_args()

    profile = load_profile(a.robot)
    u = load_robot(profile)
    if a.clip:
        d = json.load(open(a.clip, encoding="utf-8"))
        t = np.asarray(d["t"], dtype=float)
        if "," in a.frames or "." in a.frames:
            times = [float(x) for x in a.frames.split(",")]
        else:
            n = int(a.frames)
            times = list(np.linspace(t[0], t[-1], n + 2)[1:-1])
        stem = os.path.splitext(os.path.basename(a.clip))[0]
        for k, tk in enumerate(times):
            i = int(np.argmin(np.abs(t - tk)))
            joints = {j: d["data"][j][i] for j in d["joints"]}
            render_pose(u, profile, joints, os.path.join(a.out, f"clip_{stem}_{k}_t{t[i]:.2f}.png"), f"{stem} @ t={t[i]:.2f}s")
    else:
        only = set(a.only.split(",")) if a.only else None
        for name, joints in CHECK_POSES:
            if only and name not in only:
                continue
            render_pose(u, profile, joints, os.path.join(a.out, f"{name}.png"), name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
