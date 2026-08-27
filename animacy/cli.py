"""``animacy`` command line.

    animacy check <robot>                      validate a ROBOT.md (+ URDF)
    animacy profile export <robot> -o out.json  JSON for the web viewer
    animacy retarget --robot <robot> --clip <clip_dir> -o out.csv [--mode m] [--format f]
    animacy clip json <clip_dir> -o out.json    canonical clip → web JSON
    animacy capture ...                         (animacy.capture, optional deps)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _cmd_check(a) -> int:
    from .profile import find_robot

    prof = find_robot(a.robot)
    errs = prof.check()
    if errs:
        print(f"FAIL {prof.name} ({prof.path})")
        for e in errs:
            print("  -", e)
        return 1
    print(f"OK {prof.name}: {len(prof.joints)} joints, modes {list(prof.retarget)}, urdf {os.path.relpath(prof.urdf_path(), prof.dir)}")
    return 0


def _cmd_profile_export(a) -> int:
    from .profile import export_web_json, find_robot

    prof = find_robot(a.robot)
    errs = prof.check()
    if errs and not a.force:
        print("refusing to export a profile that fails `animacy check` (use --force):", *errs, sep="\n  - ")
        return 1
    out = a.output or os.path.join("web", "robots", f"{prof.name}.json")
    export_web_json(prof, out)
    print("wrote", out)
    return 0


def _cmd_retarget(a) -> int:
    from .export import write_joint_table
    from .profile import find_robot
    from .retarget import retarget_clip
    from .schema import HumanClip

    prof = find_robot(a.robot)
    clip = HumanClip.load(a.clip, audio=False)
    probs = clip.validate()
    if probs:
        print("clip problems:", *probs, sep="\n  - ")
        if not a.force:
            return 1
    table = retarget_clip(clip, prof, mode=a.mode)
    out = write_joint_table(table, prof, a.output, fmt=a.format)
    print(f"wrote {out} ({len(table)} frames, {table['t'].iloc[-1]:.2f}s, mode={a.mode}, format={a.format})")
    return 0


def _cmd_clip_json(a) -> int:
    from .schema import HumanClip

    clip = HumanClip.load(a.clip, audio=False)
    out = a.output or os.path.join(a.clip, "motion.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(clip.to_web_json(), fh)
    print("wrote", out)
    return 0


def _cmd_capture(a) -> int:
    from .capture import main as capture_main

    return capture_main(a)


def _cmd_mirror(a) -> int:
    from .mirror import run_from_args

    return run_from_args(a)


def _cmd_say(a) -> int:
    from .serve import main as say_main

    return say_main(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animacy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate a robot profile")
    c.add_argument("robot")
    c.set_defaults(fn=_cmd_check)

    pr = sub.add_parser("profile", help="profile tools")
    prs = pr.add_subparsers(dest="pcmd", required=True)
    pe = prs.add_parser("export", help="write web JSON for a robot")
    pe.add_argument("robot")
    pe.add_argument("-o", "--output")
    pe.add_argument("--force", action="store_true")
    pe.set_defaults(fn=_cmd_profile_export)

    r = sub.add_parser("retarget", help="canonical clip → robot joints")
    r.add_argument("--robot", required=True)
    r.add_argument("--clip", required=True, help="clip directory (motion.parquet)")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--mode", default="default")
    r.add_argument("--format", default="auto", help="auto|autonomous_os_csv|pollen_move|csv|json")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=_cmd_retarget)

    cl = sub.add_parser("clip", help="clip tools")
    cls_ = cl.add_subparsers(dest="ccmd", required=True)
    cj = cls_.add_parser("json", help="canonical clip → web JSON")
    cj.add_argument("clip")
    cj.add_argument("-o", "--output")
    cj.set_defaults(fn=_cmd_clip_json)

    cap = sub.add_parser("capture", help="webcam/video → canonical clip (needs mediapipe)")
    cap.add_argument("--source", default="0", help="camera index, video path, or directory of videos")
    cap.add_argument("-o", "--output", required=True, help="clip directory to write")
    cap.add_argument("--arm", default="right", choices=["right", "left", "none"])
    cap.add_argument("--duration", type=float, default=0.0, help="seconds (0 = until q/EOF)")
    cap.add_argument("--no-audio", action="store_true")
    cap.add_argument("--preview", action="store_true", help="show a window while capturing")
    cap.add_argument("--neutral-seconds", type=float, default=1.0, help="seconds of neutral pose at the start used for zeroing")
    cap.set_defaults(fn=_cmd_capture)

    s = sub.add_parser("say", help="robot speaks + moves in sync (text -> TTS -> motion source -> retarget -> robot)")
    s.add_argument("text")
    s.add_argument("--robot", required=True)
    s.add_argument("--source", default="envelope", help="model | retrieval | envelope (labelled heuristic)")
    s.add_argument("--sink", default=None, help="override runtime.kind: reachy_daemon | autonomous_os_hal | print")
    s.add_argument("--url", default=None)
    s.add_argument("--tts", default="auto", help="auto | sapi | espeak | kokoro")
    s.add_argument("--no-audio", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--checkpoint", default="checkpoints/v1")
    s.set_defaults(fn=_cmd_say)

    mi = sub.add_parser("mirror", help="live: video/webcam -> trackers -> retarget -> robot (needs mediapipe)")
    mi.add_argument("--source", default="0")
    mi.add_argument("--robot", required=True)
    mi.add_argument("--mode", default="default")
    mi.add_argument("--sink", default=None)
    mi.add_argument("--url", default=None)
    mi.add_argument("--speed", type=float, default=1.0)
    mi.add_argument("--duration", type=float, default=0.0)
    mi.add_argument("--start", type=float, default=0.0)
    mi.add_argument("--preview", action="store_true")
    mi.add_argument("--arm", default="right", choices=["right", "left", "none"])
    mi.add_argument("--neutral-seconds", type=float, default=1.0)
    mi.add_argument("--pose-every", type=int, default=1)
    mi.add_argument("--hold", type=float, default=0.5)
    mi.add_argument("--readback-every", type=float, default=5.0)
    mi.add_argument("--log", default=None)
    mi.set_defaults(fn=_cmd_mirror)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
