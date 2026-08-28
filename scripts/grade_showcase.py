"""Export a grading run's best clips per movement as clean MP4s + showcase.json (see animacy/grade/showcase.py).

    python scripts/grade_showcase.py --run data/grading/<run> --robot lamp --source retrieval --out data/grading/showcase_lamp
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.grade.showcase import export_showcase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--robot", default="lamp")
    ap.add_argument("--source", default="retrieval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-movement", type=int, default=1)
    ap.add_argument("--no-sealed", action="store_true", help="skip held-out clips entirely")
    ap.add_argument("--software", action="store_true")
    a = ap.parse_args()
    m = export_showcase(a.run, a.robot, a.out, source=a.source, per_movement=a.per_movement,
                        include_sealed=not a.no_sealed, gpu=not a.software)
    n = sum(len(v["exported"]) for v in m["movements"].values())
    print(f"[showcase] {n} clip(s) -> {os.path.abspath(a.out)}/showcase.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
