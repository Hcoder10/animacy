"""Sim-to-real calibration run on a physical Reachy Mini via the daemon HTTP API.

Builds a synthetic CANONICAL clip (look left/right/up/down, roll, brows, lean),
retargets it through robots/reachy_mini/ROBOT.md exactly like any captured
clip, streams the joint table to the robot at 30 Hz with a slew clamp, narrates
each segment aloud (Windows SAPI) so a person can judge direction, and reads
the daemon's present head pose / antenna positions back at each segment's peak
so commanded-vs-measured is logged as evidence.

    python scripts/reachy_sim2real.py --host 192.168.1.60 [--pitch-sign -1] [--dry-run]

Conventions under test (see ROBOT.md "Sign conventions"):
  canonical +head_yaw = LEFT, +head_pitch = UP, +head_roll = right ear down,
  antennas sent as [left, right] radians (reachy-duplex action_space order).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import load_profile  # noqa: E402
from animacy.retarget import retarget_clip  # noqa: E402
from animacy.schema import HumanClip, empty_frames  # noqa: E402

RATE = 30.0

# (label, {canonical channel: amplitude}, hold seconds)
SEGMENTS = [
    ("center", {}, 1.0),
    ("look left", {"head_yaw": 35.0}, 2.0),
    ("center", {}, 0.8),
    ("look right", {"head_yaw": -35.0}, 2.0),
    ("center", {}, 0.8),
    ("look up", {"head_pitch": 25.0}, 2.0),
    ("center", {}, 0.8),
    ("look down", {"head_pitch": -25.0}, 2.0),
    ("center", {}, 0.8),
    ("roll, right ear down", {"head_roll": 20.0}, 2.0),
    ("center", {}, 0.8),
    ("both brows up", {"brow_l": 1.0, "brow_r": 1.0}, 2.0),
    ("center", {}, 0.8),
    ("left brow only", {"brow_l": 1.0}, 2.0),
    ("center", {}, 0.8),
    ("lean in", {"head_x": 100.0, "torso_lean_fwd": 15.0}, 2.0),
    ("center", {}, 0.8),
    ("turn body left", {"torso_yaw": 40.0}, 2.0),
    ("center", {}, 1.0),
]


def smoothstep(n_ramp: int, n_hold: int) -> np.ndarray:
    up = np.linspace(0, 1, n_ramp)
    up = up * up * (3 - 2 * up)
    return np.concatenate([up, np.ones(n_hold), up[::-1]])


def build_clip():
    """Canonical clip + per-frame segment labels + peak frame index per segment."""
    frames, labels, peaks = [], [], []
    ramp = int(0.5 * RATE)
    for label, chans, hold in SEGMENTS:
        n_hold = int(hold * RATE)
        env = smoothstep(ramp, n_hold) if chans else np.zeros(n_hold)
        n = len(env)
        seg = empty_frames(n)
        seg["face_valid"] = 1.0
        for ch, amp in chans.items():
            seg[ch] = amp * env
        peaks.append(len(labels) + (ramp + n_hold // 2 if chans else n // 2))
        labels += [label] * n
        frames.append(seg)
    df = pd.concat(frames, ignore_index=True)
    df["t"] = np.arange(len(df)) / RATE
    return HumanClip.from_frames(df, source="synthetic", note="sim2real calibration"), labels, peaks


def say(text: str) -> None:
    safe = text.replace("'", "''")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Robot:
    def __init__(self, host: str, port: int, pitch_sign: float, dry: bool):
        self.base = f"http://{host}:{port}"
        self.s = requests.Session()
        self.pitch_sign = pitch_sign
        self.dry = dry

    def get(self, path, timeout=4.0):
        return self.s.get(self.base + path, timeout=timeout).json()

    def post(self, path, body=None, timeout=6.0):
        if self.dry:
            return {}
        r = self.s.post(self.base + path, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def prepare(self):
        print("motors:", self.get("/api/motors/status"))
        self.post("/api/motors/set_mode/enabled")
        time.sleep(0.5)
        print("motors now:", self.get("/api/motors/status"))
        print("wake_up ...")
        self.post("/api/move/play/wake_up", timeout=20)
        for _ in range(60):
            time.sleep(0.25)
            if not self.get("/api/move/running"):
                break
        print("present head pose after wake:", self.get("/api/state/present_head_pose"))

    def target(self, row) -> dict:
        return {
            "target_head_pose": {
                "x": float(row["head_x"]) / 1000.0, "y": float(row["head_y"]) / 1000.0, "z": float(row["head_z"]) / 1000.0,
                "roll": math.radians(float(row["head_roll"])),
                "pitch": self.pitch_sign * math.radians(float(row["head_pitch"])),
                "yaw": math.radians(float(row["head_yaw"])),
            },
            "target_antennas": [math.radians(float(row["antenna_left"])), math.radians(float(row["antenna_right"]))],
            "target_body_yaw": math.radians(float(row["body_yaw"])),
        }

    def set_target(self, row):
        self.post("/api/move/set_target", self.target(row), timeout=1.5)

    def read_back(self) -> dict:
        try:
            hp = self.get("/api/state/present_head_pose")
            ant = self.get("/api/state/present_antenna_joint_positions")
            by = self.get("/api/state/present_body_yaw")
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        return {"head_pose_rad_m": hp, "antennas_rad": ant, "body_yaw_rad": by}

    def goto_neutral(self, duration=1.5):
        self.post("/api/move/goto", {"head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                                     "antennas": [0.0, 0.0], "body_yaw": 0.0, "duration": duration,
                                     "interpolation": "minjerk"}, timeout=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.60")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pitch-sign", type=float, default=-1.0,
                    help="multiply canonical pitch before sending; -1 because the daemon's +pitch looks DOWN (measured Aug 2026)")
    ap.add_argument("--dry-run", action="store_true", help="retarget + print, send nothing")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sim2real"))
    a = ap.parse_args()

    prof = load_profile(os.path.join(ROOT, "robots", "reachy_mini"))
    clip, labels, peaks = build_clip()
    table = retarget_clip(clip, prof)
    # retarget_clip may stretch time; map labels by time.
    src_t = clip.frames["t"].to_numpy()
    lab_idx = np.searchsorted(src_t, table["t"].to_numpy(), side="right") - 1
    lab_idx = np.clip(lab_idx, 0, len(labels) - 1)
    table_labels = [labels[i] for i in lab_idx]
    print(f"clip {len(clip)} frames -> joint table {len(table)} frames ({table['t'].iloc[-1]:.1f}s)")
    print("joint ranges:", {j: (round(table[j].min(), 1), round(table[j].max(), 1)) for j in prof.joint_names})
    if a.dry_run:
        return 0

    robot = Robot(a.host, a.port, a.pitch_sign, dry=False)
    robot.prepare()
    log = {"host": a.host, "pitch_sign": a.pitch_sign, "started": datetime.now().isoformat(), "segments": []}
    say_thread = None
    last_label = None
    current = {}
    dt = 1.0 / RATE
    caps = {"deg": 6.0, "mm": 4.0}  # per-frame slew clamps
    try:
        for i in range(len(table)):
            row = table.iloc[i]
            label = table_labels[i]
            if label != last_label:
                if not a.no_voice:
                    if say_thread is not None:
                        say_thread.join()
                    say_thread = threading.Thread(target=say, args=(label,), daemon=True)
                    say_thread.start()
                print(f"[{row['t']:6.2f}s] {label}", flush=True)
                seg = {"label": label, "t": float(row["t"]), "commanded_peak": None, "measured_peak": None}
                log["segments"].append(seg)
                last_label = label
            # slew clamp
            cmd = {}
            for j in prof.joints:
                v = float(row[j.name])
                if j.name in current:
                    cap = caps["mm"] if j.unit == "mm" else (caps["deg"] * 3 if "antenna" in j.name else caps["deg"])
                    v = current[j.name] + max(-cap, min(cap, v - current[j.name]))
                cmd[j.name] = v
            current = cmd
            t0 = time.perf_counter()
            try:
                robot.set_target(cmd)
            except Exception as e:  # noqa: BLE001
                print("  set_target error:", e)
            # peak read-back: the middle of the hold
            seg = log["segments"][-1]
            src_i = int(np.searchsorted(src_t, row["t"], side="right") - 1)
            if seg["measured_peak"] is None and any(src_i >= p for p in peaks if labels[min(p, len(labels) - 1)] == label):
                seg["commanded_peak"] = {k: round(v, 2) for k, v in cmd.items()}
                seg["measured_peak"] = robot.read_back()
                hp = seg["measured_peak"].get("head_pose_rad_m", {})
                if hp:
                    print(f"    measured: yaw {math.degrees(hp['yaw']):6.1f} pitch {math.degrees(hp['pitch']):6.1f} "
                          f"roll {math.degrees(hp['roll']):6.1f} z {hp['z']*1000:5.1f}mm | antennas "
                          f"{[round(math.degrees(x),1) for x in seg['measured_peak']['antennas_rad']]} | body "
                          f"{math.degrees(seg['measured_peak']['body_yaw_rad']):6.1f}")
            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        try:
            robot.goto_neutral()
        except Exception as e:  # noqa: BLE001
            print("goto_neutral failed:", e)
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"reachy_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1)
        print("log ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
