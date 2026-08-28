"""Print what the current cut actually is: shot list, screen-time balance,
camera usage, transitions and anything still missing.

    python scripts/video/edit_summary.py            # the summary
    python scripts/video/edit_summary.py --shots    # plus every shot

Reads data/video/edit/edl.json, so run edit_all.py (even with --no-render)
first. This is the thing to look at before deciding the cut is finished.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import load_edl, mmss, EDL_JSON  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", action="store_true", help="print every shot")
    args = ap.parse_args()

    if not EDL_JSON.exists():
        print(f"no EDL at {EDL_JSON} - run scripts/video/edit_all.py first")
        return 1
    edl = load_edl()
    shots = sorted(edl.shots, key=lambda s: s.start)

    if args.shots:
        print(f"{'start':>8} {'dur':>6}  {'kind':<8} {'label':<18} source")
        for s in shots:
            src = Path(s.src).name if s.src else "-"
            note = f"   [{s.note}]" if s.note else ""
            print(f"{s.start:>8.2f} {s.dur:>6.2f}  {s.kind:<8} {s.label:<18} {src}{note}")
        print()

    film = edl.total
    by_kind: dict[str, float] = {}
    for s in shots:
        by_kind[s.kind] = by_kind.get(s.kind, 0.0) + s.dur
    print(f"runtime      {mmss(film)}  ({len(shots)} shots, {len(edl.lines)} lines)")
    for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"  {k:<9} {v:>6.1f}s  {100 * v / max(film, 1e-9):>5.1f}%")

    cams: dict[str, list[float]] = {}
    for s in shots:
        if s.kind == "cam":
            cams.setdefault(s.label, []).append(s.dur)
    print("\ncameras")
    for k in sorted(cams):
        v = cams[k]
        print(f"  {k}  {sum(v):>6.1f}s over {len(v)} shot(s)")

    # the thing that makes a cut feel like a slideshow: a long stretch with no
    # human (or robot) face in it
    runs: list[list] = []
    for s in shots:
        k = "broll" if s.kind == "broll" else "other"
        if runs and runs[-1][0] == k:
            runs[-1][1] += s.dur
            runs[-1][2] += 1
        else:
            runs.append([k, s.dur, 1, s.start])
    longest = sorted((r for r in runs if r[0] == "broll"), key=lambda r: -r[1])[:4]
    print("\nlongest unbroken b-roll stretches (watch these for slideshow feel)")
    for _k, d, n, st in longest:
        print(f"  {d:>5.1f}s across {n} shot(s), from {mmss(st)}")

    print(f"\ntransitions  {len(edl.dissolves)} dissolve(s) at "
          f"{[mmss(d['at']) for d in edl.dissolves]}, "
          f"{max(0, len(shots) - 1 - len(edl.dissolves))} hard cut(s)")
    print(f"lower-thirds {[t.text for t in edl.titles]}")

    if edl.dropped:
        print(f"\ndropped for length ({len(edl.dropped)}):")
        for d in edl.dropped:
            print(f"  - {d}")

    slugs = [s for s in shots if s.kind == "slug"]
    if slugs:
        print(f"\nSTILL MISSING - {len(slugs)} placeholder(s) in the cut:")
        for s in slugs:
            print(f"  {s.label} section {s.section} at {mmss(s.start)} ({s.dur:.1f}s)")
    else:
        print("\nno placeholders: every shot is real footage")

    notes = [s for s in shots if s.note]
    if notes:
        print(f"\nshots carrying a note ({len(notes)}):")
        for s in notes:
            print(f"  {mmss(s.start):>6}  {s.label:<18} {s.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
