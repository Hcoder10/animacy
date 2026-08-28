"""Add `black_windows` to every manifest entry, without re-recording anything.

    python scripts/video/broll_annotate_black.py

A cut that lands inside a black stretch goes black mid-sentence. The grading
reels legitimately contain them (they are the spacers between the judged clips),
so the fix is to publish where they are rather than remove them. `register()`
now records this for new clips; this backfills the ones already made.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import MANIFEST, OUT_DIR, black_windows, log  # noqa: E402


def main() -> int:
    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    n_with = 0
    for c in man["clips"]:
        path = os.path.join(OUT_DIR, c["file"])
        if not os.path.exists(path):
            log(f"  {c['file']}: missing, skipped")
            continue
        wins = black_windows(path)
        c["black_windows"] = wins
        if wins:
            n_with += 1
            log(f"  {c['file']}: " + ", ".join(f"{w['start']}-{w['end']}s" for w in wins))
    man["black_windows_note"] = (
        "black_windows lists stretches of (near) black in each clip, as "
        "[{start, end, seconds}] in clip time. Choose an in-point that avoids them, or "
        "shorten the shot. Empty list = the clip is clean all the way through.")
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    log(f"\n{len(man['clips'])} clips scanned, {n_with} contain black; manifest updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
