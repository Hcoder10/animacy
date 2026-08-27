"""Render a robot joint table to an MP4 through the web viewer (headless).

    python web/dev/render_clip.py --robot lamp --native nod --out out/nod.mp4
    python web/dev/render_clip.py --robot reachy_mini --table data/grading/<run>/clips/<clip>.json --out out/x.mp4
    python web/dev/render_clip.py --robot lamp --csv some_autonomous_recording.csv --audio say.wav --title "Clip 3" \\
        --subtitle "The robot says: hi" --out out/x.mp4

Inputs: a vendor native clip (``--native``), an Autonomous OS CSV (``--csv``),
or a joint-table JSON (``--table``: ``{"t": [...], "data": {joint: [...]}}``,
the shape written by ``animacy.export`` ``json`` and by the grading run).
The video is 30 fps, fixed 3/4 front camera, no UI, optional title card and
16 kHz WAV muxed in. Implementation: ``animacy/grade/render.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from animacy.export import read_autonomous_os_csv  # noqa: E402
from animacy.grade.movements import load_vendor_table  # noqa: E402
from animacy.grade.render import ViewerRenderer  # noqa: E402
from animacy.profile import find_robot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--native", help="vendor clip name in robots/<robot>/clips/native")
    src.add_argument("--csv", help="Autonomous OS CSV (timestamp, <joint>.pos)")
    src.add_argument("--table", help="joint-table JSON ({t, data})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--card-seconds", type=float, default=1.0)
    ap.add_argument("--audio", default=None, help="16 kHz mono WAV to mux (starts after the card)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--software", action="store_true", help="SwiftShader instead of the GPU")
    ap.add_argument("--zoom", type=float, default=0.9)
    a = ap.parse_args()

    profile = find_robot(a.robot)
    if a.native:
        table = load_vendor_table(profile, a.native, a.max_seconds)
    elif a.csv:
        table = read_autonomous_os_csv(a.csv)
    else:
        obj = json.load(open(a.table, encoding="utf-8"))
        t = np.asarray(obj["t"], dtype=np.float64)
        table = pd.DataFrame({"t": t - t[0], **{k: np.asarray(v, dtype=np.float64) for k, v in obj["data"].items()}})
    if a.max_seconds is not None:
        table = table[table["t"] <= a.max_seconds].reset_index(drop=True)
    audio, sr = None, 16000
    if a.audio:
        import soundfile as sf

        data, sr = sf.read(a.audio, dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
    with ViewerRenderer(gpu=not a.software, zoom=a.zoom) as r:
        info = r.render_clip(a.robot, table, profile, a.out, title=a.title, subtitle=a.subtitle, audio=audio, sr=sr,
                             card_seconds=a.card_seconds if (a.title or a.subtitle) else 0.0)
    print(json.dumps({"out": os.path.abspath(a.out), **info}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
