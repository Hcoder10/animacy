"""Ambient life on a real robot from the growing human corpus.

Every few seconds: pick a random clip in data/clips (re-scanned each round, so
new captures join automatically; dropped clips excluded via _index.json), cut a
random 3-5 s segment of real human motion, retarget it through the robot's
ROBOT.md and stream it to the robot; then settle and wait.

    python scripts/reachy_ambient.py --robot reachy_mini --url http://192.168.1.60:8000 [--gap 5] [--min 3 --max 5]
Stop: Ctrl-C, or `python scripts/reachy_ambient.py --stop` (writes a stop file).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.profile import find_robot  # noqa: E402
from animacy.retarget import retarget_clip  # noqa: E402
from animacy.schema import HumanClip  # noqa: E402
from animacy.sinks import make_sink, stream_table  # noqa: E402

STOP_FILE = os.path.join(ROOT, "data", "ambient.stop")


def kept_clips(clips_dir: str):
    idx = os.path.join(clips_dir, "_index.json")
    dropped = set()
    if os.path.exists(idx):
        try:
            d = json.load(open(idx, encoding="utf-8"))
            rows = d.get("clips", d) if isinstance(d, dict) else d
            for r in rows:
                if isinstance(r, dict) and r.get("status") == "dropped":
                    dropped.add(r.get("name"))
        except Exception:  # noqa: BLE001
            pass
    out = []
    for n in sorted(os.listdir(clips_dir)):
        p = os.path.join(clips_dir, n)
        if n in dropped or not os.path.exists(os.path.join(p, "motion.parquet")):
            continue
        out.append(p)
    return out


def random_segment(clip: HumanClip, min_s: float, max_s: float, rng: random.Random):
    f = clip.frames
    valid = f["face_valid"].to_numpy() > 0
    n = len(f)
    seg_len = int(rng.uniform(min_s, max_s) * clip.rate_hz)
    if n <= seg_len + 2:
        return None
    for _ in range(20):
        start = rng.randrange(0, n - seg_len)
        if valid[start:start + seg_len].mean() > 0.95:
            sub = f.iloc[start:start + seg_len].copy()
            sub["t"] = np.arange(len(sub)) / clip.rate_hz
            # re-zero on the segment's own first frames so it starts near rest
            for c in ("head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "torso_yaw", "torso_lean_fwd", "torso_lean_side"):
                sub[c] = sub[c] - float(sub[c].iloc[:5].mean())
            return HumanClip.from_frames(sub, source="ambient", rate_hz=clip.rate_hz)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="reachy_mini")
    ap.add_argument("--url", default=None)
    ap.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--gap", type=float, default=5.0, help="seconds between segments")
    ap.add_argument("--min", type=float, default=3.0)
    ap.add_argument("--max", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stop", action="store_true", help="signal a running loop to stop")
    a = ap.parse_args()
    if a.stop:
        open(STOP_FILE, "w").write("stop")
        print("stop requested")
        return 0
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    rng = random.Random(a.seed)
    prof = find_robot(a.robot)
    sink = make_sink(prof, None, a.url)
    sink.prepare()
    print(f"[ambient] {prof.name} via {getattr(sink, 'base', sink.name)}; corpus {a.clips}; gap {a.gap}s", flush=True)
    n_played = 0
    try:
        while not os.path.exists(STOP_FILE):
            clips = kept_clips(a.clips)
            if not clips:
                time.sleep(a.gap)
                continue
            path = rng.choice(clips)
            try:
                clip = HumanClip.load(path, audio=False)
                seg = random_segment(clip, a.min, a.max, rng)
                if seg is None:
                    continue
                table = retarget_clip(seg, prof)
                n_played += 1
                print(f"[ambient] #{n_played} {os.path.basename(path)} {seg.duration:.1f}s -> {len(table)} frames ({len(clips)} clips in corpus)", flush=True)
                stream_table(table, prof, sink)
            except Exception as e:  # noqa: BLE001
                print("[ambient] skip:", type(e).__name__, e, flush=True)
            time.sleep(a.gap)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sink.neutral(1.0)
        except Exception:  # noqa: BLE001
            pass
        print("[ambient] stopped after", n_played, "segments", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
