"""Batch-import every browser recording (zip or segment dir) under a folder.

    python scripts/import_browser_clips.py data/browser --out data/clips [--no-smooth]

Each ``*.zip`` becomes ``<out>/<zipstem>/`` (one segment) or
``<out>/<zipstem>/<segment>/`` (several); each plain sub-directory containing
``*motion.json`` files is imported the same way. Prints one summary line per
clip and exits non-zero if any clip fails ``HumanClip.validate()``.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.import_browser import find_segments, import_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src_dir")
    p.add_argument("--out", default=os.path.join("data", "clips"))
    p.add_argument("--no-smooth", action="store_true")
    a = p.parse_args()

    jobs = []
    for f in sorted(os.listdir(a.src_dir)):
        full = os.path.join(a.src_dir, f)
        if f.lower().endswith(".zip"):
            jobs.append((full, os.path.join(a.out, os.path.splitext(f)[0])))
        elif os.path.isdir(full) and find_segments(full):
            jobs.append((full, os.path.join(a.out, f)))
    if not jobs:
        print(f"nothing to import under {a.src_dir}")
        return 1
    rc, total = 0, 0
    for src, out in jobs:
        print(f"== {src}")
        r, dirs = import_path(src, out, smooth=not a.no_smooth)
        rc |= r
        total += len(dirs)
    print(f"{total} clip dir(s) written under {a.out}; rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
