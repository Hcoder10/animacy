#!/usr/bin/env python
"""Pollen's Reachy Mini emotion/dance library -> animacy joint tables (json).

Two sources are understood, with identical channel semantics:

* ``--json-dir``: Pollen's recorded moves as shipped in
  ``pollen-robotics/reachy-mini-emotions-library`` and ``-dances-library``
  (Apache-2.0): ``{"description", "time": [...], "set_target_data": [{"head":
  4x4, "antennas": [right, left], "body_yaw": rad}]}`` sampled at ~50 Hz.
  Preferred when available (4x the temporal resolution of the npz).
* ``--npz-dir``: reachy-duplex's conversion of the same files
  (``training/reachy_library_to_motion.py``): ``raw`` = 9 columns
  ``[x, y, z (m), roll, pitch, yaw (deg), antennas[0], antennas[1] (rad), body_yaw (rad)]``
  at ``fps`` = 12.5 Hz. Note reachy-duplex labels the antenna columns
  ``antenna_l, antenna_r`` but the SDK order is ``[right, left]``
  (``reachy_mini`` daemon joint order ``right_antenna, left_antenna``); we use
  the SDK order.

Conversion to animacy / ROBOT.md units and signs (see robots/reachy_mini/urdf/README.md):

    head_x/y/z     = translation * 1000                (mm; SDK world frame, x fwd / y left / z up)
    head_roll      = +roll                             (deg; + = right ear down, same as canonical)
    head_pitch     = -pitch                            (deg; SDK/ROS +pitch = nose DOWN, animacy + = UP)
    head_yaw       = +yaw                              (deg; + = left)
    body_yaw       = deg(body_yaw)                     (+ = left)
    antenna_left   = +deg(antennas[1])                 (+ = swings outward/down, away from the midline)
    antenna_right  = -deg(antennas[0])                 (mirror axis on the real robot; flipped so + = outward too)

Euler angles are the SDK's own convention (``create_head_pose``:
``Rotation.from_euler("xyz")`` = Rz(yaw) Ry(pitch) Rx(roll)); decomposition uses
``Rotation.as_euler("xyz")``.

Output: ``robots/reachy_mini/clips/native/<name>.json`` in the format of
``animacy.export.write_joint_table(fmt="json")`` plus ``description`` /
``source`` keys, resampled to 30 Hz, clamped to ROBOT.md joint limits (the
fraction of clamped samples is recorded), rounded to 2 decimals; and
``index.json``.

    python scripts/pollen_npz_to_joints.py                 # curated set (~16 clips)
    python scripts/pollen_npz_to_joints.py --all
    python scripts/pollen_npz_to_joints.py --names yes1,no1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import load_profile  # noqa: E402

RDIR = os.path.join(ROOT, "robots", "reachy_mini")
OUT_DIR = os.path.join(RDIR, "clips", "native")
DEFAULT_JSON_DIR = r"C:\Users\sarta\reachy-duplex\data\reachy_lib"
DEFAULT_NPZ_DIR = r"C:\Users\sarta\reachy-duplex\data\motion_reachy"
RATE_HZ = 30.0
JOINTS = ["head_x", "head_y", "head_z", "head_roll", "head_pitch", "head_yaw", "body_yaw", "antenna_left", "antenna_right"]

CURATED = [
    "amazed1", "attentive1", "boredom1",
    "yes1", "simple_nod", "no1", "sad1", "laughing1", "curious1", "surprised1",
    "proud1", "thoughtful1", "welcoming1", "confused1", "cheerful1", "head_tilt_roll",
]

SOURCE_NOTE = ("Pollen Robotics reachy-mini-emotions-library / reachy-mini-dances-library "
               "(Apache-2.0, https://huggingface.co/pollen-robotics), converted by scripts/pollen_npz_to_joints.py")


# ---------------------------------------------------------------------------
def raw_from_json(path: str):
    d = json.load(open(path, encoding="utf-8"))
    t = np.asarray(d["time"], dtype=np.float64)
    rows = []
    for e in d["set_target_data"]:
        M = np.asarray(e["head"], dtype=np.float64)
        rpy = R.from_matrix(M[:3, :3]).as_euler("xyz", degrees=True)
        ant = e.get("antennas") or [0.0, 0.0]
        rows.append([M[0, 3], M[1, 3], M[2, 3], rpy[0], rpy[1], rpy[2], ant[0], ant[1], e.get("body_yaw") or 0.0])
    return t, np.asarray(rows, dtype=np.float64), d.get("description", "")


def raw_from_npz(path: str):
    d = np.load(path, allow_pickle=True)
    raw = np.asarray(d["raw"], dtype=np.float64)
    fps = float(d["fps"])
    t = np.arange(len(raw)) / fps
    return t, raw, str(d["description"]) if "description" in d else ""


def raw_to_animacy(raw: np.ndarray) -> dict:
    """9 SDK columns -> animacy joints (mm / deg, animacy signs)."""
    return {
        "head_x": raw[:, 0] * 1000.0,
        "head_y": raw[:, 1] * 1000.0,
        "head_z": raw[:, 2] * 1000.0,
        "head_roll": raw[:, 3],
        "head_pitch": -raw[:, 4],
        "head_yaw": raw[:, 5],
        "body_yaw": np.degrees(raw[:, 8]),
        "antenna_left": np.degrees(raw[:, 7]),
        "antenna_right": -np.degrees(raw[:, 6]),
    }


def resample(t: np.ndarray, cols: dict, rate_hz: float):
    t = t - t[0]
    n = int(np.floor(t[-1] * rate_hz + 1e-9)) + 1
    tn = np.arange(n) / rate_hz
    return tn, {k: np.interp(tn, t, v) for k, v in cols.items()}


def convert(name: str, t, raw, description, profile, source_file: str) -> tuple[dict, dict]:
    cols = raw_to_animacy(raw)
    tn, cols = resample(t, cols, RATE_HZ)
    clipped = 0
    total = 0
    data = {}
    for j in profile.joints:
        v = cols[j.name]
        c = np.clip(v, j.min, j.max)
        clipped += int(np.sum(np.abs(c - v) > 1e-9))
        total += len(v)
        data[j.name] = [round(float(x) + 0.0, 2) for x in c]
    clip = {
        "robot": profile.name,
        "rate_hz": RATE_HZ,
        "joints": profile.joint_names,
        "t": [round(float(x), 4) for x in tn],
        "data": data,
        "description": description,
        "source": source_file,
    }
    entry = {"name": name, "seconds": round(float(tn[-1]), 2), "frames": len(tn), "description": description,
             "source_file": source_file, "clamped_fraction": round(clipped / max(total, 1), 4)}
    return clip, entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help="Pollen library JSON root (searched recursively)")
    ap.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR, help="reachy-duplex npz dir (fallback)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--names", help="comma-separated clip names (default: curated set)")
    ap.add_argument("--all", action="store_true", help="convert every clip found")
    ap.add_argument("--prefer-npz", action="store_true", help="use the npz even when the JSON source exists")
    a = ap.parse_args()

    profile = load_profile(RDIR)
    assert profile.joint_names == JOINTS, profile.joint_names

    sources = {}  # name -> (kind, path)
    if not a.prefer_npz and os.path.isdir(a.json_dir):
        for p in glob.glob(os.path.join(a.json_dir, "**", "*.json"), recursive=True):
            sources[os.path.splitext(os.path.basename(p))[0]] = ("json", p)
    if os.path.isdir(a.npz_dir):
        for p in glob.glob(os.path.join(a.npz_dir, "*.npz")):
            sources.setdefault(os.path.splitext(os.path.basename(p))[0], ("npz", p))
    if not sources:
        print("no sources found", file=sys.stderr)
        return 1

    if a.all:
        names = sorted(sources)
    elif a.names:
        names = a.names.split(",")
    else:
        names = CURATED
    missing = [n for n in names if n not in sources]
    if missing:
        print("WARNING: not found:", missing)
        names = [n for n in names if n in sources]

    os.makedirs(a.out, exist_ok=True)
    index = []
    total_bytes = 0
    for n in names:
        kind, path = sources[n]
        t, raw, desc = raw_from_json(path) if kind == "json" else raw_from_npz(path)
        if len(t) < 2:
            print("skip", n, "(too short)")
            continue
        rel = os.path.relpath(path, os.path.dirname(a.json_dir if kind == "json" else a.npz_dir)).replace("\\", "/")
        clip, entry = convert(n, t, raw, desc, profile, f"{kind}:{rel}")
        out = os.path.join(a.out, f"{n}.json")
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(clip, fh, separators=(",", ":"))
        total_bytes += os.path.getsize(out)
        index.append(entry)
        print(f"  {n:24s} {entry['seconds']:6.2f}s {entry['frames']:4d} frames  from {kind}  clamped {entry['clamped_fraction']:.3f}")

    idx = {
        "robot": profile.name,
        "rate_hz": RATE_HZ,
        "format": "json (animacy.export.write_joint_table fmt=json) + description/source",
        "units": {"head_x": "mm", "head_y": "mm", "head_z": "mm", "head_roll": "deg", "head_pitch": "deg",
                  "head_yaw": "deg", "body_yaw": "deg", "antenna_left": "deg", "antenna_right": "deg"},
        "conventions": {
            "frame": "x forward, y robot-left, z up; head pose relative to the robot base (independent of body_yaw)",
            "head_pitch": "+ = look UP (negated from the SDK, whose +pitch is nose-down)",
            "head_yaw": "+ = turn left", "head_roll": "+ = right ear down", "body_yaw": "+ = body turns left",
            "antennas": "+ = swings outward/down away from the head's midline; 0 = vertical. "
                        "SDK values: antennas=[right, left] rad, right_sdk = -antenna_right, left_sdk = +antenna_left",
        },
        "source": SOURCE_NOTE,
        "license": "Apache-2.0",
        "clips": index,
    }
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(idx, fh, indent=1)
    print(f"wrote {len(index)} clips, {total_bytes / 1024:.0f} KB total, index -> {os.path.join(a.out, 'index.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
