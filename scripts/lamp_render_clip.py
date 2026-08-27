"""Render the Autonomous Lamp URDF at keyframes of vendor clips, to PNG.

Loads ``robots/lamp/urdf/lamp.urdf`` with yourdfpy, converts a vendor CSV to
URDF joint values through the animacy profile
(``animacy.export.read_autonomous_os_csv`` -> ``animacy.retarget.to_urdf_values``),
and rasterises every link mesh with a small software renderer (matplotlib
PolyCollection, painter's algorithm, Lambert shading) so it works headless on
Windows without OpenGL. Each panel is a 3/4 view plus a pure side view, with
the joint pivots (blue dots), the head's look direction (magenta line) and the
joint values printed.

    python scripts/lamp_render_clip.py            # standard keyframe set + contact sheet
    python scripts/lamp_render_clip.py --clip nod --frames 0,12,20 --out out/dir
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import yourdfpy  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from animacy.export import read_autonomous_os_csv  # noqa: E402
from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import to_urdf_values  # noqa: E402

ROBOT_DIR = os.path.join(ROOT, "robots", "lamp")
URDF = os.path.join(ROBOT_DIR, "urdf", "lamp.urdf")
CLIPS = os.path.join(ROBOT_DIR, "clips", "native")
PREVIEW = os.path.join(ROBOT_DIR, "urdf", "preview")
JOINTS = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]
# look direction of the head (unit, head-link frame): normal of the light disc in the CAD
LOOK_HEAD = np.array([0.70, 0.0, -0.71]) / np.linalg.norm([0.70, 0.0, -0.71])
LOOK_ORIGIN_HEAD = np.array([0.060, 0.0, -0.030])  # light-disc centre relative to the head pivot


class LampModel:
    def __init__(self, urdf_path: str = URDF):
        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        self.meshes = {}
        base_dir = os.path.dirname(os.path.abspath(urdf_path))
        for lname, link in self.urdf.link_map.items():
            vis = link.visuals[0]
            path = os.path.normpath(os.path.join(base_dir, vis.geometry.mesh.filename))
            self.meshes[lname] = trimesh.load(path, force="mesh")

    def set_deg(self, q_deg: dict):
        self.urdf.update_cfg({k: math.radians(float(v)) for k, v in q_deg.items()})

    def set_rad(self, q_rad: dict):
        self.urdf.update_cfg({k: float(v) for k, v in q_rad.items()})

    def link_T(self, link: str) -> np.ndarray:
        return self.urdf.get_transform(link, "base")

    def pivots(self) -> dict:
        return {j.name: self.link_T(j.child)[:3, 3] for j in self.urdf.actuated_joints}

    def head_look(self):
        T = self.link_T("head")
        o = T[:3, :3] @ LOOK_ORIGIN_HEAD + T[:3, 3]
        d = T[:3, :3] @ LOOK_HEAD
        return o, d


def _camera(az_deg: float, el_deg: float):
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # target -> camera
    f = -d
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def draw(ax, model: LampModel, az=-50, el=18, target=(0.0, 0.0, 0.2), half=0.27, title="", light=(-0.4, 0.6, 0.7)):
    """Orthographic software render of the current pose into ``ax``."""
    r, u, f = _camera(az, el)
    light = np.asarray(light, float)
    light /= np.linalg.norm(light)
    tgt = np.asarray(target, float)
    polys, depths, colors = [], [], []
    # ground disk
    ang = np.linspace(0, 2 * math.pi, 48)
    g = np.c_[0.25 * np.cos(ang), 0.25 * np.sin(ang), np.zeros_like(ang)] - tgt
    polys.append(np.c_[g @ r, g @ u])
    depths.append(-1e3)
    colors.append((0.90, 0.90, 0.90, 1))
    for lname, mesh in model.meshes.items():
        T = model.link_T(lname)
        v = mesh.vertices @ T[:3, :3].T + T[:3, 3] - tgt
        n = mesh.face_normals @ T[:3, :3].T
        faces = mesh.faces
        # no back-face culling: several vendor solids (the base drum) have inconsistent winding
        n = np.where((n @ f)[:, None] > 0, -n, n)  # flip normals to face the camera for shading
        tri = v[faces]  # (F,3,3)
        depth = (tri @ f).mean(axis=1)
        lam = np.clip(n @ light, 0, 1)
        base = np.array([0.93, 0.90, 0.84]) if lname != "neck" else np.array([0.45, 0.45, 0.45])
        shade = (0.35 + 0.65 * lam)[:, None] * base[None, :]
        px = tri @ r
        py = tri @ u
        polys.extend(np.stack([px, py], axis=-1))
        depths.extend(depth)
        colors.extend(np.c_[shade, np.ones(len(shade))])
    order = np.argsort(np.asarray(depths))  # small depth along f = far from camera -> draw first
    pc = PolyCollection([polys[i] for i in order], facecolors=[colors[i] for i in order], edgecolors="none", antialiased=False)
    ax.add_collection(pc)
    # forward arrow (+x), joint pivots, look direction
    a0 = np.array([0.0, 0.0, 0.002]) - tgt
    a1 = np.array([0.22, 0.0, 0.002]) - tgt
    ax.plot([a0 @ r, a1 @ r], [a0 @ u, a1 @ u], color="tab:red", lw=1.5)
    ax.text(a1 @ r, a1 @ u, " +x (front)", color="tab:red", fontsize=7, va="center")
    for p in model.pivots().values():
        q = p - tgt
        ax.plot(q @ r, q @ u, "o", ms=3.5, color="tab:blue", mec="k", mew=0.4, zorder=5)
    o, d = model.head_look()
    p0 = o - tgt
    p1 = o + 0.12 * d - tgt
    ax.plot([p0 @ r, p1 @ r], [p0 @ u, p1 @ u], color="magenta", lw=2, zorder=6)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)


def pose_panel(fig, gs_cells, model, q_deg, label):
    """One keyframe = 3/4 view + side view (camera on -y, so +x is to the right)."""
    model.set_deg(q_deg)
    hz = model.link_T("head")[:3, 3]
    txt = " ".join(f"{k[:5]}={q_deg[k]:+.0f}" for k in JOINTS)
    ax1 = fig.add_subplot(gs_cells[0])
    draw(ax1, model, az=-50, el=18, title=f"{label}  (3/4 view)")
    ax2 = fig.add_subplot(gs_cells[1])
    draw(ax2, model, az=-90, el=0, title=f"side  head pivot z={hz[2]:.2f} m  x={hz[0]:+.2f} m")
    ax2.text(0.02, 0.02, txt, transform=ax2.transAxes, fontsize=6.5, ha="left", va="bottom", family="monospace")


def clip_table(name: str, profile):
    df = read_autonomous_os_csv(os.path.join(CLIPS, f"{name}.csv"))
    urdf_vals = to_urdf_values(df, profile)  # radians, keyed by urdf joint name
    return df, urdf_vals


def lowest_head_frame(model, urdf_vals) -> int:
    """Frame whose FK puts the head pivot lowest."""
    n = len(next(iter(urdf_vals.values())))
    z = []
    for i in range(n):
        model.set_rad({j: v[i] for j, v in urdf_vals.items()})
        z.append(model.link_T("head")[2, 3])
    return int(np.argmin(z))


def extreme(df, key):
    """Index of the frame maximising ``key(row)``."""
    return int(np.argmax([key(df.iloc[i]) for i in range(len(df))]))


STANDARD = [
    # (file stem, clip, frame chooser, label)
    ("rest_idle_first", "idle", lambda df: 0, "REST: idle first frame"),
    ("zero_pose", None, None, "vendor zero (all joints 0)"),
    ("sad_extreme", "sad", "lowest_head", "SAD: extreme (lowest head)"),
    ("sad_end", "sad", lambda df: len(df) - 1, "SAD: end frame"),
    ("nod_extreme", "nod", lambda df: extreme(df, lambda r: r["elbow_pitch"]), "NOD: extreme"),
    ("headshake_min", "headshake", lambda df: extreme(df, lambda r: -r["wrist_roll"]), "HEADSHAKE: roll min"),
    ("headshake_max", "headshake", lambda df: extreme(df, lambda r: r["wrist_roll"]), "HEADSHAKE: roll max"),
    ("stretching_extreme", "stretching", lambda df: extreme(df, lambda r: r["base_pitch"] + r["elbow_pitch"]), "STRETCHING: extreme"),
    ("curious_extreme", "curious", lambda df: extreme(df, lambda r: abs(r["base_yaw"]) + abs(r["wrist_roll"])), "CURIOUS: extreme"),
    ("sleepy_extreme", "sleepy", "lowest_head", "SLEEPY: extreme (lowest head)"),
    ("wake_up_first", "wake_up", lambda df: 0, "WAKE_UP: first frame (asleep)"),
]


def render_standard(out_dir: str, model: LampModel, profile):
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for stem, clip, chooser, label in STANDARD:
        if clip is None:
            q = {k: 0.0 for k in JOINTS}
            label2 = label
        else:
            df, urdf_vals = clip_table(clip, profile)
            i = lowest_head_frame(model, urdf_vals) if chooser == "lowest_head" else chooser(df)
            # through the profile path (what the viewer/retarget use), then back to deg for the label
            q = {k: math.degrees(float(urdf_vals[profile.joint(k).urdf_joint][i])) for k in JOINTS}
            label2 = f"{label} [frame {i}/{len(df) - 1}, t={df['t'].iloc[i]:.2f}s]"
        fig = plt.figure(figsize=(8, 4.2))
        gs = fig.add_gridspec(1, 2, wspace=0.02, left=0.01, right=0.99, top=0.9, bottom=0.08)
        pose_panel(fig, [gs[0, 0], gs[0, 1]], model, q, label2)
        path = os.path.join(out_dir, f"{stem}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        items.append((label2, q))
        print("wrote", path)
    # contact sheet
    n = len(items)
    cols = 2
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(16, 4.0 * rows))
    gs = fig.add_gridspec(rows, cols * 2, wspace=0.02, hspace=0.25, left=0.01, right=0.99, top=0.97, bottom=0.02)
    for k, (label2, q) in enumerate(items):
        rr, cc = divmod(k, cols)
        pose_panel(fig, [gs[rr, 2 * cc], gs[rr, 2 * cc + 1]], model, q, label2)
    path = os.path.join(out_dir, "contact_sheet.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", help="vendor clip name (clips/native/<name>.csv)")
    ap.add_argument("--frames", help="comma-separated frame indices (default: first, most-displaced, last)")
    ap.add_argument("--out", default=PREVIEW)
    a = ap.parse_args(argv)
    profile = load_profile(ROBOT_DIR)
    model = LampModel()
    if not a.clip:
        render_standard(a.out, model, profile)
        return 0
    df, urdf_vals = clip_table(a.clip, profile)
    if a.frames:
        idx = [int(x) for x in a.frames.split(",")]
    else:
        q0 = df[JOINTS].iloc[0].to_numpy()
        idx = [0, int(np.argmax(np.linalg.norm(df[JOINTS].to_numpy() - q0, axis=1))), len(df) - 1]
    os.makedirs(a.out, exist_ok=True)
    fig = plt.figure(figsize=(8, 4.2 * len(idx)))
    gs = fig.add_gridspec(len(idx), 2, wspace=0.02, hspace=0.25, left=0.01, right=0.99, top=0.96, bottom=0.03)
    for k, i in enumerate(idx):
        q = {kk: math.degrees(float(urdf_vals[profile.joint(kk).urdf_joint][i])) for kk in JOINTS}
        pose_panel(fig, [gs[k, 0], gs[k, 1]], model, q, f"{a.clip} frame {i}/{len(df) - 1} t={df['t'].iloc[i]:.2f}s")
    path = os.path.join(a.out, f"{a.clip}_frames.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
