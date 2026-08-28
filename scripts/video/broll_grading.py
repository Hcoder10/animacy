"""B-roll for section 8: a slice of an actual blind grading reel.

    python scripts/video/broll_grading.py [--run 20260827_1501_run3] [--movement excitement]

The reel is exactly the file the judge was given: numbered "Clip N" cards with
the spoken line and nothing else — no robot name, no source, no method. That
anonymity is the point, so the cards stay in frame. Offsets come from the run's
own sealed manifest and the reel builder's own card/gap constants, so the window
lands on a card boundary rather than being eyeballed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from broll_common import FPS, H, OUT_DIR, ROOT, W, ffmpeg_bin, log, register  # noqa: E402

sys.path.insert(0, ROOT)
from animacy.grade.reel import CARD_SECONDS, GAP_SECONDS  # noqa: E402


def entries(manifest: dict, robot: str, reel_index: int, per_reel: int) -> list[dict]:
    """The clips on one reel, in order, with their start offset inside it."""
    clips = manifest["robots"][robot]["clips"]
    ordered = [clips[k] for k in sorted(clips, key=lambda s: int(s))]
    chunk = ordered[reel_index * per_reel:(reel_index + 1) * per_reel]
    t = 0.0
    for i, c in enumerate(chunk):
        c["_start"] = t
        c["_card_end"] = t + CARD_SECONDS
        c["_end"] = t + CARD_SECONDS + c["duration"] + GAP_SECONDS
        c["_n"] = reel_index * per_reel + i + 1
        t = c["_end"]
    return chunk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260827_1501_run3")
    ap.add_argument("--robot", default="lamp")
    ap.add_argument("--movement", default="excitement",
                    help="prefer a window starting on this movement's card")
    ap.add_argument("--min-seconds", type=float, default=7.0)
    ap.add_argument("--max-seconds", type=float, default=14.0)
    ap.add_argument("--out", default="s8_grading_reel.mp4")
    a = ap.parse_args()

    run_dir = os.path.join(ROOT, "data", "grading", a.run)
    with open(os.path.join(run_dir, "manifest_sealed.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    reels = sorted(f for f in os.listdir(os.path.join(run_dir, "reels"))
                   if f.startswith(f"{a.robot}_reel") and f.endswith(".mp4"))
    total = len(man["robots"][a.robot]["clips"])
    per_reel = -(-total // len(reels))
    log(f"  {total} clips for {a.robot} across {len(reels)} reels ({per_reel} per reel)")

    pick = None
    for ri, reel in enumerate(reels):
        for c in entries(man, a.robot, ri, per_reel):
            if c["movement"] != a.movement:
                continue
            # take this card + clip, plus the next entry if the pair still fits
            chunk = entries(man, a.robot, ri, per_reel)
            after = [x for x in chunk if x["_start"] >= c["_end"]]
            end = c["_end"]
            if end - c["_start"] < a.min_seconds and after:
                end = min(after[0]["_end"], c["_start"] + a.max_seconds)
            length = min(end - c["_start"], a.max_seconds)
            if length >= a.min_seconds:
                pick = (reel, c, length)
                break
        if pick:
            break
    if not pick:
        raise SystemExit(f"no {a.movement} window of {a.min_seconds}-{a.max_seconds} s found")

    reel, c, length = pick
    src = os.path.join(run_dir, "reels", reel)
    # start a beat before the card so the cut lands on the black between clips
    start = max(0.0, c["_start"] - 0.35)
    log(f"  {reel}: Clip {c['_n']} ({c['movement']}, {c['line_set']} lines) at "
        f"{c['_start']:.2f} s; taking {length:.2f} s from {start:.2f} s")
    log(f"  card reads: Clip {c['_n']} / {c['card_line']}")

    out_path = os.path.join(OUT_DIR, a.out)
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
                    "-i", src, "-t", f"{length:.3f}",
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1117,fps={FPS}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", out_path], check=True)
    register(a.out, section="8",
             shows=f"A slice of the reel the blind judge actually watched: the numbered "
                   f"\"Clip {c['_n']}\" card with only the spoken line, then the motion. The card "
                   f"never says which robot, which source or which method — that anonymity is the "
                   f"test. Movement: {c['movement']} ({c['line_set']} lines).",
             source=f"data/grading/{a.run}/reels/{reel}, {start:.2f}-{start + length:.2f} s "
                    f"(offsets from manifest_sealed.json + reel.CARD_SECONDS/GAP_SECONDS)",
             notes="The reel is 512x512, so the frame is the reel pillarboxed on the film's "
                   "background. The clip's own identity stays sealed: only the card is shown.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
