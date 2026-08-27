"""``animacy preview``: render a robot's calibration poses to PNG *through its
ROBOT.md mapping* and print a sign probe, so a headless coding agent can read
directions without a browser.

    animacy preview robots/<name> [--out <dir>] [--poses rest,look_left,...] [--clip <clip>]
    python -m animacy.preview robots/so101            # same thing without the cli hook

Pipeline per pose: canonical channels (a calibration pose: "look left" = head_yaw
+40, ...) -> ``retarget.raw_joint_targets`` (the ROBOT.md mapping) ->
``to_urdf_values`` (urdf_sign/offset/units) -> yourdfpy FK -> matplotlib 3D
(single painter's-sorted Poly3DCollection, Lambert shading), three views
(3/4 from front-left, front from +x, side from +y), axes labelled, joint values
printed on the image, magenta ray = the tip link's pointing direction.

``--clip`` accepts a canonical clip directory (``motion.parquet``; retargeted
with ``--mode``), an animacy joint-table ``.json`` (``write_joint_table``
fmt=json) or an Autonomous OS ``.csv``; ``--frames N`` keyframes are rendered.

The probe (also written to ``<out>/probe.txt`` and ``probe.json``): for every
joint, +10 units from ``rest`` -> tip displacement (mm) and gaze yaw/pitch
change (deg). Tip link / gaze axis are auto-detected (``urdf_tools``) or set
with ``--tip`` / ``--gaze`` or ``ROBOT.md description.viewer.tip_link/gaze``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .profile import Profile, find_robot
from .urdf_tools import (format_probe, joints_from_channels, load_urdf, pick_gaze_axis, pick_tip_link, probe,
                         rest_joints, scene_triangles, set_joints, set_table_row, tip_state)

# name -> (mode, canonical channel values). Everything unspecified is 0 = neutral.
POSES: "OrderedDict[str, tuple]" = OrderedDict([
    ("rest", ("default", {})),
    ("look_left", ("default", {"head_yaw": 40.0})),
    ("look_up", ("default", {"head_pitch": 25.0})),
    ("roll", ("default", {"head_roll": 20.0})),
    ("brows", ("default", {"brow_l": 1.0, "brow_r": 1.0})),
    ("lean_in", ("default", {"torso_lean_fwd": 25.0, "head_x": 80.0})),
    ("mouth", ("default", {"mouth_open": 1.0})),
    ("puppet_wave", ("puppet", {"shoulder_yaw": 15.0, "shoulder_pitch": 100.0, "elbow_flex": 80.0,
                                "wrist_pitch": 10.0, "hand_open": 0.9})),
])
DEFAULT_POSES = ["rest", "look_left", "look_up", "roll", "brows", "lean_in", "puppet_wave"]
VIEWS = (("3/4 from front-left", (25, 35)), ("front (camera on +x)", (10, 0)), ("side (camera on +y)", (10, 90)))
DISPLAY_FACES = 900
PALETTE = [(0.40, 0.40, 0.43), (0.85, 0.55, 0.20), (0.90, 0.72, 0.30), (0.20, 0.45, 0.85), (0.93, 0.93, 0.93),
           (0.85, 0.20, 0.20), (0.30, 0.65, 0.35), (0.60, 0.40, 0.75), (0.55, 0.55, 0.60), (0.25, 0.30, 0.40)]


class Previewer:
    def __init__(self, profile: Profile, tip: Optional[str] = None, gaze: Optional[str] = None, display_faces: int = DISPLAY_FACES):
        self.profile = profile
        self.u = load_urdf(profile, load_meshes=True, display_faces=display_faces)
        self.tip = pick_tip_link(profile, self.u, tip)
        self.gaze, self.gaze_label = pick_gaze_axis(profile, self.u, self.tip, gaze)
        self.colors = {}
        set_joints(self.u, profile, rest_joints(profile))
        self.limits = self._limits()

    def _limits(self):
        b = self.u.scene.bounds
        if b is None:
            return (-0.3, 0.3), (-0.3, 0.3), (0.0, 0.6)
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        c = (lo + hi) / 2
        ext = float(np.max(hi - lo))
        half = 0.62 * ext + 0.03
        zlo = min(0.0, lo[2]) - 0.01
        return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half), (zlo, max(hi[2], zlo + 0.05) + 0.35 * ext)

    def color(self, link: str):
        if link not in self.colors:
            self.colors[link] = PALETTE[len(self.colors) % len(PALETTE)]
        return self.colors[link]

    def draw(self, ax, view, title: str):
        from matplotlib.colors import LightSource
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tris = scene_triangles(self.u)
        if tris:
            faces = np.concatenate([f for f, _ in tris], axis=0)
            base = np.concatenate([np.tile(np.asarray(self.color(l))[None, :], (len(f), 1)) for f, l in tris], axis=0)
            n = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
            shade = LightSource(azdeg=315, altdeg=45).shade_normals(n, fraction=1.0)
            cols = np.clip(base * (0.45 + 0.55 * shade[:, None]), 0, 1)
            ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="none"))
        (x0, x1), (y0, y1), (z0, z1) = self.limits
        L = 0.15 * (x1 - x0)
        for d, c in (((L, 0, 0), "r"), ((0, L, 0), "g"), ((0, 0, L), "b")):
            ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color=c, lw=2.5)
        o, g = tip_state(self.u, self.tip, self.gaze)
        R = 0.3 * (x1 - x0)
        ax.plot([o[0], o[0] + R * g[0]], [o[1], o[1] + R * g[1]], [o[2], o[2] + R * g[2]], color="m", lw=3)
        ax.scatter([o[0]], [o[1]], [o[2]], color="m", s=20)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_zlim(z0, z1)
        ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_xlabel("x fwd")
        ax.set_ylabel("y left")
        ax.set_zlabel("z up")
        ax.set_title(title, fontsize=9)

    def render(self, joints: Dict[str, float], out_png: str, label: str, subtitle: str = "") -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        set_joints(self.u, self.profile, joints)
        fig = plt.figure(figsize=(15, 5.4))
        for i, (t, v) in enumerate(VIEWS):
            self.draw(fig.add_subplot(1, 3, i + 1, projection="3d"), v, t)
        jtxt = ", ".join(f"{j.name}={joints.get(j.name, j.rest):.1f}{'' if j.unit == 'unit' else j.unit}" for j in self.profile.joints)
        fig.suptitle(f"{self.profile.name}: {label}   {subtitle}".rstrip(), fontsize=10)
        fig.text(0.5, 0.015, jtxt, ha="center", fontsize=8.5, family="monospace")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        fig.savefig(out_png, dpi=80)
        plt.close(fig)
        return out_png


# ---------------------------------------------------------------- clip loading
def load_clip_table(profile: Profile, path: str, mode: str) -> pd.DataFrame:
    """Canonical clip dir -> retargeted joint table; .json/.csv joint tables as-is."""
    if os.path.isdir(path):
        from .retarget import retarget_clip
        from .schema import HumanClip

        clip = HumanClip.load(path, audio=False)
        return retarget_clip(clip, profile, mode=mode)
    if path.lower().endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        return pd.DataFrame({"t": d["t"], **{j: d["data"][j] for j in d["joints"]}})
    if path.lower().endswith(".csv"):
        from .export import read_autonomous_os_csv

        return read_autonomous_os_csv(path)
    raise ValueError(f"--clip must be a clip directory, a joint-table .json or an autonomous_os .csv: {path}")


def frame_indices(t: np.ndarray, frames: str) -> List[int]:
    if "," in frames or "." in frames:
        return [int(np.argmin(np.abs(t - float(x)))) for x in frames.split(",")]
    n = max(1, int(frames))
    return [int(i) for i in np.round(np.linspace(0, len(t) - 1, n + 2)[1:-1])] if n < len(t) else list(range(len(t)))


# ---------------------------------------------------------------- driver
def run(robot: str, out: Optional[str] = None, poses: Optional[Sequence[str]] = None, clip: Optional[str] = None,
        frames: str = "3", mode: str = "default", tip: Optional[str] = None, gaze: Optional[str] = None,
        quiet: bool = False) -> Dict:
    profile = find_robot(robot)
    out = out or os.path.join(profile.dir, "urdf", "preview")
    pv = Previewer(profile, tip=tip, gaze=gaze)
    result = {"robot": profile.name, "tip": pv.tip, "gaze": pv.gaze_label, "out": out, "pngs": [], "skipped": []}

    rows = probe(profile, pv.u, pv.tip, pv.gaze)
    text = format_probe(profile, rows, pv.tip, pv.gaze_label)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "probe.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out, "probe.json"), "w", encoding="utf-8") as fh:
        json.dump({"robot": profile.name, "tip": pv.tip, "gaze": pv.gaze_label, "rows": rows}, fh, indent=1)
    result["probe"] = rows
    if not quiet:
        print(text)

    if clip is None:
        names = list(poses) if poses else list(DEFAULT_POSES)
        for name in names:
            if name not in POSES:
                raise ValueError(f"unknown pose {name!r}; known: {list(POSES)}")
            pmode, channels = POSES[name]
            if pmode not in profile.retarget:
                result["skipped"].append(f"{name} (no `{pmode}` mode in ROBOT.md)")
                continue
            joints, _ = joints_from_channels(profile, pmode, channels)
            sub = "[%s] " % pmode + ", ".join(f"{k}={v:g}" for k, v in channels.items()) if channels else "[%s] all channels 0" % pmode
            png = pv.render(joints, os.path.join(out, f"{name}.png"), name, sub)
            set_joints(pv.u, profile, joints)
            p, g = tip_state(pv.u, pv.tip, pv.gaze)
            result["pngs"].append(png)
            if not quiet:
                print(f"wrote {png}  tip {np.round(p, 3).tolist()} gaze {np.round(g, 2).tolist()}  " +
                      ", ".join(f"{k}={v:.1f}" for k, v in joints.items()))
    else:
        table = load_clip_table(profile, clip, mode)
        t = table["t"].to_numpy(dtype=float)
        stem = os.path.splitext(os.path.basename(os.path.normpath(clip)))[0]
        for i in frame_indices(t, frames):
            joints = set_table_row(pv.u, profile, table, i)
            png = pv.render(joints, os.path.join(out, f"clip_{stem}_t{t[i]:.2f}.png"), f"{stem} @ t={t[i]:.2f}s", f"[{mode}]")
            result["pngs"].append(png)
            if not quiet:
                print("wrote", png)
    if result["skipped"] and not quiet:
        print("skipped:", "; ".join(result["skipped"]))
    return result


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("robot", help="robots/<name> directory or robot name")
    p.add_argument("--out", help="output directory (default robots/<name>/urdf/preview)")
    p.add_argument("--poses", default=",".join(DEFAULT_POSES), help="comma-separated; known: " + ",".join(POSES))
    p.add_argument("--clip", help="canonical clip dir | joint-table .json | autonomous_os .csv (renders keyframes instead of poses)")
    p.add_argument("--frames", default="3", help="with --clip: N evenly spaced frames, or comma-separated times (s)")
    p.add_argument("--mode", default="default", help="retarget mode for a canonical --clip")
    p.add_argument("--tip", help="tip/face link (default: viewer.tip_link, else auto)")
    p.add_argument("--gaze", help="pointing axis in the tip frame: x|y|z|-x|-y|-z or vx,vy,vz (default: viewer.gaze, else auto)")
    p.add_argument("--quiet", action="store_true")


def main(a=None) -> int:
    if a is None or isinstance(a, list):
        ap = argparse.ArgumentParser(prog="animacy preview", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        add_arguments(ap)
        a = ap.parse_args(a)
    poses = [s for s in a.poses.split(",") if s] if a.poses else None
    run(a.robot, out=a.out, poses=poses, clip=a.clip, frames=a.frames, mode=a.mode, tip=a.tip, gaze=a.gaze, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
