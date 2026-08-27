"""Robot joint tables → vendor-native files.

Formats:

* ``autonomous_os_csv`` — Autonomous OS ``hal/recordings/*.csv``: a
  ``timestamp`` column (seconds) plus ``<joint>.pos`` columns, 30 Hz. Exactly
  what ``POST /servo/upload`` on a Lamp accepts and ``/servo/play`` plays.
  :func:`validate_autonomous_os_csv` mirrors the server's own checks so a file
  that passes here is accepted there.
* ``pollen_move`` — Pollen Robotics recorded-move JSON (``reachy_mini`` emotion
  library layout): ``{"description", "time": [...], "set_target_data": [{"head":
  4x4, "antennas": [l, r], "body_yaw": f}]}`` with head pose in metres/radians.
* ``csv`` / ``json`` — plain tables for anything else.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .profile import Profile

_JOINT_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.pos$")
AUTONOMOUS_MAX_ROWS = 20000          # server: _MAX_SERVO_RECORDING_ROWS (kept conservative)
AUTONOMOUS_MAX_BYTES = 5 * 1024 * 1024


def _fmt_for(profile: Profile, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if profile.export.formats:
        return profile.export.formats[0]
    return "csv"


def to_autonomous_os_csv(table: pd.DataFrame, profile: Profile) -> str:
    cfg = profile.export.autonomous_os_csv or {}
    suffix = cfg.get("column_suffix", ".pos")
    tcol = cfg.get("timestamp_column", "timestamp")
    fps = float(cfg.get("fps", profile.rate_hz))
    t = np.arange(len(table)) / fps
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    joints = [j.name for j in profile.joints]
    w.writerow([tcol] + [f"{j}{suffix}" for j in joints])
    for i in range(len(table)):
        w.writerow([f"{t[i]:.6f}"] + [f"{float(table[j].iloc[i]):.6f}" for j in joints])
    return buf.getvalue()


def validate_autonomous_os_csv(text: str, valid_joints: Optional[List[str]] = None,
                               max_speed: Optional[Dict[str, float]] = None) -> List[str]:
    """Mirror of ``hal/routes/servo.py:upload_servo_recording`` plus the
    SAFETY.md speed ceiling. Empty list = the Lamp would accept this file."""
    errs: List[str] = []
    if not text:
        return ["empty csv"]
    if len(text.encode("utf-8")) > AUTONOMOUS_MAX_BYTES:
        errs.append("csv too large")
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if "timestamp" not in fields:
        errs.append('missing required column "timestamp"')
        return errs
    joint_fields = [f for f in fields if f != "timestamp"]
    if not joint_fields:
        errs.append("missing joint columns (expected *.pos fields)")
        return errs
    bad = [f for f in joint_fields if not _JOINT_FIELD_RE.match(f)]
    if bad:
        errs.append(f"invalid joint columns: {bad}. Expected <name>.pos")
    if valid_joints is not None:
        want = {f"{j}.pos" for j in valid_joints}
        unknown = [f for f in joint_fields if f not in want]
        if unknown:
            errs.append(f"unknown joint columns: {unknown}. Valid: {sorted(want)}")
    rows = list(reader)
    if len(rows) > AUTONOMOUS_MAX_ROWS:
        errs.append(f"too many rows (max {AUTONOMOUS_MAX_ROWS})")
    prev_t, prev = None, None
    for i, row in enumerate(rows):
        try:
            t = float(row["timestamp"])
        except Exception:
            errs.append(f"invalid timestamp at row {i + 2}")
            break
        vals = {}
        for f in joint_fields:
            v = row.get(f)
            if v is None or v == "":
                errs.append(f"missing value for {f} at row {i + 2}")
                break
            try:
                vals[f] = float(v)
            except Exception:
                errs.append(f"invalid float for {f} at row {i + 2}")
                break
        if max_speed and prev is not None and t > prev_t:
            for f, v in vals.items():
                cap = max_speed.get(f[:-4])
                if cap and abs(v - prev[f]) / (t - prev_t) > cap * 1.05:
                    errs.append(f"{f} exceeds max_speed {cap}/s at row {i + 2}")
                    break
        prev_t, prev = t, vals
        if errs:
            break
    return errs


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def to_pollen_move(table: pd.DataFrame, profile: Profile, description: str = "") -> Dict:
    """Reachy Mini recorded-move JSON. Expects joints named head_x/y/z (mm),
    head_roll/pitch/yaw (deg), antenna_left/right (deg), body_yaw (deg)."""
    need = ["head_x", "head_y", "head_z", "head_roll", "head_pitch", "head_yaw", "antenna_left", "antenna_right", "body_yaw"]
    missing = [n for n in need if n not in table.columns]
    if missing:
        raise ValueError(f"pollen_move needs joints {missing}")
    data = []
    for i in range(len(table)):
        r = table.iloc[i]
        M = np.eye(4)
        M[:3, :3] = _rpy_to_matrix(math.radians(r["head_roll"]), math.radians(r["head_pitch"]), math.radians(r["head_yaw"]))
        M[:3, 3] = [r["head_x"] / 1000.0, r["head_y"] / 1000.0, r["head_z"] / 1000.0]
        data.append({
            "head": M.tolist(),
            "antennas": [math.radians(r["antenna_left"]), math.radians(r["antenna_right"])],
            "body_yaw": math.radians(r["body_yaw"]),
        })
    return {"description": description or f"animacy retarget for {profile.name}",
            "time": [float(x) for x in table["t"]], "set_target_data": data}


def write_joint_table(table: pd.DataFrame, profile: Profile, out_path: str, fmt: str = "auto") -> str:
    fmt = _fmt_for(profile, fmt)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if fmt == "autonomous_os_csv":
        text = to_autonomous_os_csv(table, profile)
        errs = validate_autonomous_os_csv(text, profile.joint_names, {j.name: j.max_speed for j in profile.joints})
        if errs:
            raise ValueError("autonomous_os_csv would be rejected by the Lamp: " + "; ".join(errs))
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    elif fmt == "pollen_move":
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(to_pollen_move(table, profile), fh)
    elif fmt == "csv":
        table.to_csv(out_path, index=False, float_format="%.6f")
    elif fmt == "json":
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"robot": profile.name, "rate_hz": profile.rate_hz, "joints": profile.joint_names,
                       "t": table["t"].round(4).tolist(),
                       "data": {j: table[j].round(3).tolist() for j in profile.joint_names}}, fh)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return out_path


def read_autonomous_os_csv(path: str) -> pd.DataFrame:
    """Vendor CSV → joint table (``t`` + joint names without ``.pos``)."""
    d = pd.read_csv(path)
    out = pd.DataFrame({"t": d["timestamp"] - d["timestamp"].iloc[0]})
    for c in d.columns:
        if c != "timestamp":
            out[c[:-4] if c.endswith(".pos") else c] = d[c]
    return out
