"""Are the host renders consistent with the timeline the edit is cutting to?

The narration timings come from data/video/podcast/show.json and the host
motion is baked into the camera renders against that same clock. When show.json
is regenerated, every camera clip made against the previous version is stale:
it is exactly as long as the old timeline and drifts against the narration by
the difference. Nothing about a still frame reveals it.

    python scripts/video/edit_check_sources.py           # one report
    python scripts/video/edit_check_sources.py --watch   # until consistent

Exit code 0 means every camera clip matches the current show.json and the edit
can be rendered. This is cheap enough to poll, unlike a full build.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_common import (  # noqa: E402
    PODCAST_DIR, SHOW_JSON, CAMERAS, CUTPLAN, probe, _cam_from_path, _sec_num,
    _infer_section_and_base, _probe_cache,
)

# The (camera, section) pairs the cut plan actually asks for. A stale clip for
# an angle this film never cuts to is not a reason to hold up a render.
WANTED: set[tuple[str, int]] = {
    (label, section)
    for section, plan in CUTPLAN.items()
    for spec in plan for label in spec
    if label in CAMERAS
}

FULL_TOL = 0.6        # a full-timeline clip must match the show this closely
SECTION_MIN = -0.6    # a section clip may be this much shorter than its section
SECTION_MAX = 2.5     # ...or this much longer, for a lead-in or tail handle


def check() -> tuple[bool, list[str]]:
    _probe_cache.clear()
    notes: list[str] = []
    if not SHOW_JSON.exists():
        return False, [f"no {SHOW_JSON}"]
    blob = json.loads(SHOW_JSON.read_text(encoding="utf-8"))
    show = float(blob.get("seconds") or 0.0)
    sec_t0, sec_dur = {}, {}
    for s in blob.get("sections", []):
        n = _sec_num(s.get("number", s.get("index")))
        if n is not None:
            sec_t0[n] = float(s["t_start"])
            sec_dur[n] = float(s["t_end"]) - float(s["t_start"])

    notes.append(f"show.json: {show:.3f}s, {len(sec_t0)} sections, "
                 f"placeholder_voice={blob.get('placeholder_voice')}")

    clips = sorted(PODCAST_DIR.glob("**/*.mp4"))
    if not clips:
        return False, notes + ["no camera clips at all"]

    # Judge by slot, not by file. A superseded clip left on disk next to its
    # re-render is harmless - the build prefers the consistent, higher-versioned
    # one - so it is only a problem when nothing good covers that slot.
    unreadable: list[str] = []
    good_slots: set[tuple[str, object]] = set()
    stale_slots: dict[tuple[str, object], str] = {}
    ok: list[str] = []

    for p in clips:
        cam = _cam_from_path(p)
        if cam not in CAMERAS:
            continue
        rel = f"{cam}/{p.name}"
        d = probe(p)["dur"]
        if d <= 0.3:
            unreadable.append(rel)
            continue
        if show and abs(d - show) <= FULL_TOL:
            good_slots.add((cam, "full"))
            ok.append(f"{rel} full timeline {d:.2f}s")
            continue
        if show and d >= 0.85 * show:
            stale_slots.setdefault(
                (cam, "full"),
                f"{rel} looks full-length at {d:.2f}s but the show is {show:.2f}s")
            continue
        got = _infer_section_and_base(p, sec_t0, sec_dur)
        if not got:
            stale_slots.setdefault((cam, p.name), f"{rel} cannot be placed in any section")
            continue
        n, _base = got
        slack = d - sec_dur.get(n, d)
        if slack < SECTION_MIN or slack > SECTION_MAX:
            stale_slots.setdefault(
                (cam, n),
                f"{rel} is {d:.2f}s but section {n} is {sec_dur.get(n, 0):.2f}s "
                f"({slack:+.2f}s)")
        else:
            good_slots.add((cam, n))
            ok.append(f"{rel} section {n} {d:.2f}s ({slack:+.2f}s handle)")

    uncovered = {k: v for k, v in stale_slots.items() if k not in good_slots}
    superseded = len(stale_slots) - len(uncovered)

    # camera A is rendered as one continuous file and stands in for any angle
    # the cut plan wants but has not got, so a full-timeline clip covers all
    covers_all = {c for c, s in good_slots if s == "full"}
    blocking = {k: v for k, v in uncovered.items()
                if isinstance(k[1], int) and k in WANTED and k[0] not in covers_all}
    ignorable = {k: v for k, v in uncovered.items() if k not in blocking}

    notes.append(f"{len(ok)} clip(s) consistent, {len(blocking)} slot(s) the cut needs "
                 f"and cannot have, {len(ignorable)} stale slot(s) the cut does not use, "
                 f"{superseded} superseded file(s) ignored, "
                 f"{len(unreadable)} still being written")
    for s in blocking.values():
        notes.append(f"  BLOCKING   {s}")
    for s in ignorable.values():
        notes.append(f"  stale, unused by the cut: {s}")
    for u in unreadable:
        notes.append(f"  WRITING    {u}")
    missing = [c for c in CAMERAS if not any(s[0] == c for s in good_slots)]
    if missing:
        notes.append(f"  no usable footage at all for camera(s) {missing}")
    return (not blocking and not unreadable and not missing), notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true",
                    help="poll until every clip is consistent")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=2400.0)
    args = ap.parse_args()

    deadline = time.time() + args.timeout
    while True:
        good, notes = check()
        print("\n".join(notes), flush=True)
        if good:
            print("SOURCES CONSISTENT - safe to render", flush=True)
            return 0
        if not args.watch:
            return 1
        if time.time() > deadline:
            print("TIMED OUT waiting for consistent sources", flush=True)
            return 2
        print(f"--- waiting {args.interval:.0f}s ---", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
