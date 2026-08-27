"""Blind motion grading run (the acceptance gate). See docs/GRADING.md.

    python scripts/grade_run.py --robots lamp reachy_mini --sources model retrieval envelope --seeds 2 \\
        --out data/grading/<timestamp>

Exit code 0 only if every robot passes the gate.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.grade.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
