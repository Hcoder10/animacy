"""Inventory of data/clips for the motion model: duration, face-valid seconds, speaking
fraction, subject, and whether the mouth actually moves (a capture-quality proxy).

    python scripts/model_clips_inventory.py [data/clips]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "data/clips"
    total = 0.0
    subjects = {}
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        name = os.path.basename(os.path.dirname(d))
        mp = os.path.join(d, "motion.parquet")
        if not os.path.exists(mp):
            print(f"{name:34s} NO motion.parquet (still being written?)")
            continue
        t = pq.read_table(mp).to_pandas()
        meta_path = os.path.join(d, "meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        fv = np.nan_to_num(t["face_valid"].to_numpy())
        sp = np.nan_to_num(t["speaking"].to_numpy())
        n = len(t)
        valid = float(fv.mean() * n / 30.0)
        total += valid
        subj = str(meta.get("subject") or "-")
        subjects[subj] = subjects.get(subj, 0.0) + valid
        audio = "y" if os.path.exists(os.path.join(d, "audio.wav")) else "NO"
        print(f"{name:34s} {n / 30:6.0f}s  face_valid {fv.mean():.2f}  valid {valid:5.0f}s  speaking {sp.mean():.2f}  "
              f"audio={audio}  subject {subj}  mouth_std {np.nanstd(t['mouth_open']):.3f}  src={meta.get('source')}")
    print(f"total valid: {total / 60:.1f} min over {len(subjects)} subjects: " + ", ".join(f"{k}={v / 60:.1f}min" for k, v in sorted(subjects.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
