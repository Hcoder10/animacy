"""Prove the cut is in sync with its sources, frame by frame.

For every host and b-roll shot in the EDL, this pulls one frame out of the
rendered master and the frame it is supposed to have come from, and compares
them. A mismatch means the source offset for that shot is wrong - which is the
one failure mode that is invisible in a still and obvious the moment anyone
watches the film, because the robot's mouth stops matching the words.

    python scripts/video/edit_verify_sync.py
    python scripts/video/edit_verify_sync.py --shots 8   # sample fewer

Reports mean absolute pixel difference per shot on a 0-255 scale. Matching
frames land in the low single digits (the master has been through an extra
h264 encode); anything above ~14 is a different frame.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import load_edl, ff, mmss  # noqa: E402
from edit_render import MASTER  # noqa: E402

MATCH = 14.0     # mean abs diff below this is the same frame, re-encoded


def grab(src: Path, t: float, dst: Path) -> Path | None:
    ff(["-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
        "-vf", "scale=320:180", "-pix_fmt", "rgb24", str(dst)], check=False)
    return dst if dst.exists() and dst.stat().st_size > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=0, help="sample only N shots")
    ap.add_argument("--into", type=float, default=0.5,
                    help="seconds into each shot to sample")
    args = ap.parse_args()

    if not MASTER.exists():
        print(f"no master at {MASTER} - render first")
        return 1
    edl = load_edl()
    shots = [s for s in sorted(edl.shots, key=lambda s: s.start)
             if s.kind in ("cam", "broll") and s.src and Path(s.src).exists()]
    if args.shots:
        step = max(1, len(shots) // args.shots)
        shots = shots[::step]

    # a shot that dissolves in is a blend of two pictures for its first d
    # seconds, so sampling inside that window would compare a blend against a
    # clean source and report a false mismatch
    dis = {round(d["at"], 3): d.get("actual", d["dur"]) for d in edl.dissolves}

    tmp = Path(tempfile.mkdtemp(prefix="animacy_sync_"))
    bad, checked = [], 0
    print(f"{'at':>7}  {'shot':<18} {'diff':>6}  source")
    for i, s in enumerate(shots):
        skip = dis.get(round(s.start, 3), 0.0) + 0.15
        into = max(skip, args.into)
        into = min(into, max(0.05, s.dur - 0.2))
        a = grab(MASTER, s.start + into, tmp / f"m{i}.png")
        b = grab(Path(s.src), s.src_in + into, tmp / f"s{i}.png")
        if not a or not b:
            continue
        ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
        ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
        if ia.shape != ib.shape:
            h = min(ia.shape[0], ib.shape[0])
            w = min(ia.shape[1], ib.shape[1])
            ia, ib = ia[:h, :w], ib[:h, :w]
        # drop the bottom fifth: that is where the lower-thirds live, and they
        # are in the master by design but not in the source
        keep = int(ia.shape[0] * 0.8)
        diff = float(np.abs(ia[:keep] - ib[:keep]).mean())
        checked += 1
        flag = "" if diff <= MATCH else "   <-- MISMATCH"
        if diff > MATCH:
            bad.append((s, diff))
        print(f"{mmss(s.start):>7}  {s.label:<18} {diff:>6.2f}  "
              f"{Path(s.src).name} @ {s.src_in:.2f}s{flag}")

    print(f"\n{checked - len(bad)}/{checked} shots match their source frame")
    if bad:
        print("MISMATCHED - the source offset for these shots is wrong:")
        for s, d in bad:
            print(f"  {mmss(s.start)} {s.label} (diff {d:.1f}) "
                  f"{Path(s.src).name} @ {s.src_in:.2f}s")
        return 2
    print("every sampled shot is pulling the frame the EDL says it should")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
