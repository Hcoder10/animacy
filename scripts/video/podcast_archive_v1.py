"""Move the superseded v1 podcast footage out of the way, once the master is cut.

The show was rebuilt mid-production, so two generations of footage exist at
different lengths: the v1 clips are on the 227.0 s timeline, while show.json,
narration.wav and the ``*_v2.mp4`` clips are on 223.967 s. Mixing them puts the
picture up to three seconds ahead of its own dialogue — invisible in a still,
obvious the moment anyone watches.

This moves (never deletes) the v1 clips plus ``show_v1.json`` and
``render_manifest_v1.json`` into ``_superseded_v1/``, keeping the evidence trail
while making it impossible to pick a wrong-clock file by accident.

    python scripts/video/podcast_archive_v1.py            # dry run, prints the plan
    python scripts/video/podcast_archive_v1.py --apply    # actually move

Run it only once the edit's final master has rendered: until then the editor may
still be reading the v1 files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "data", "video", "podcast")
ARCHIVE = os.path.join(OUT, "_superseded_v1")

README = """\
Superseded footage — do not use.

These clips are on the OLD 227.0 s timeline (see show_v1.json). The film ships on
show.json / narration.wav / the *_v2.mp4 clips, which are 223.967 s. Cutting one
of these against the current show.json puts the picture up to three seconds ahead
of its own dialogue. Kept only as an evidence trail.
"""


def keep_set():
    """Absolute paths of every clip the CURRENT manifest references."""
    man = json.load(open(os.path.join(OUT, "render_manifest.json"), encoding="utf-8"))
    return {os.path.normcase(os.path.abspath(os.path.join(ROOT, c["path"].replace("/", os.sep))))
            for c in man.get("clips", [])}


def plan():
    """(source, destination) for everything that must move.

    What to KEEP is decided by the current manifest, never by the filename. An
    earlier version of this kept anything matching ``_v2``, which would have
    archived a later re-render — ``B/s01_v3.mp4`` — as if it were superseded.
    The manifest is the only thing that actually knows what the film uses."""
    keep = keep_set()
    moves = []
    for cam in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, cam)
        if not os.path.isdir(d) or cam.startswith("_") or cam in ("stills", "placeholder_voice"):
            continue
        for f in sorted(os.listdir(d)):
            src = os.path.join(d, f)
            if f.endswith(".mp4") and os.path.normcase(os.path.abspath(src)) not in keep:
                moves.append((src, os.path.join(ARCHIVE, cam, f)))
    for f in ("show_v1.json", "render_manifest_v1.json"):
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            moves.append((p, os.path.join(ARCHIVE, f)))
    return moves


def guard():
    """Refuse to run if the current build is not intact — never strand the edit."""
    problems = []
    for f in ("show.json", "narration.wav", "render_manifest.json"):
        if not os.path.exists(os.path.join(OUT, f)):
            problems.append(f"missing {f}")
    man_path = os.path.join(OUT, "render_manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
        clips = man.get("clips", [])
        if not clips:
            problems.append("render_manifest.json lists no clips")
        show = json.load(open(os.path.join(OUT, "show.json"), encoding="utf-8"))
        for c in clips:
            if not os.path.exists(os.path.join(ROOT, c["path"].replace("/", os.sep))):
                problems.append(f"current clip missing: {c['path']}")
            # the real test is the clock, not the filename: a clip that does not
            # fit the current show is from a different generation
            if c["f_end"] > show["n_frames"] or c["t_end"] > show["seconds"] + 1e-6:
                problems.append(f"manifest clip does not fit the current timeline: {c['path']}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="actually move (default is a dry run)")
    args = ap.parse_args(argv)

    problems = guard()
    if problems:
        print("refusing to archive — the current build is not intact:")
        for p in problems:
            print("  -", p)
        return 2

    moves = plan()
    print(f"{len(moves)} item(s) to move into {os.path.relpath(ARCHIVE, ROOT)}:")
    for src, dst in moves:
        print(f"  {os.path.relpath(src, OUT)}  ->  {os.path.relpath(dst, OUT)}")
    if not args.apply:
        print("\ndry run — nothing moved. Re-run with --apply once the master has rendered.")
        return 0

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    os.makedirs(ARCHIVE, exist_ok=True)
    with open(os.path.join(ARCHIVE, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(README)
    left = [c["path"] for c in json.load(open(os.path.join(OUT, "render_manifest.json"), encoding="utf-8"))["clips"]]
    missing = [p for p in left if not os.path.exists(os.path.join(ROOT, p.replace("/", os.sep)))]
    if missing:
        print("ERROR: the archive removed a clip the current manifest needs:", missing)
        return 1
    print(f"\nmoved {len(moves)} item(s); all {len(left)} current clips still present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
