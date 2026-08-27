"""Export captured clips as a LeRobot v3.0 dataset for one robot.

    python scripts/export_lerobot.py --robot lamp --clips data/clips --out data/lerobot/animacy_lamp \
        [--fps 30] [--mode default] [--exclude a,b] [--env-state audio|human|none] \
        [--validate] [--push squaredcuber/animacy-lamp-lerobot] [--private] [--force]

``--validate`` loads the written tree with the real ``LeRobotDataset`` in the
separate lerobot venv (``.venv-lerobot`` or ``$ANIMACY_LEROBOT_PYTHON``);
``--push`` uploads to the Hub only after that validation passes. See
``docs/LEROBOT.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.lerobot_export import (  # noqa: E402
    ENV_STATE_CHOICES,
    default_lerobot_python,
    export,
    push_to_hub,
    validate_with_lerobot,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="robot name under robots/ or a path to a ROBOT.md")
    ap.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"), help="directory of clip dirs (or one clip dir)")
    ap.add_argument("--out", required=True, help="output dataset root, e.g. data/lerobot/animacy_lamp")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mode", default="default", help="retarget mode from ROBOT.md")
    ap.add_argument("--exclude", default="", help="comma-separated clip names to skip")
    ap.add_argument("--min-seconds", type=float, default=3.0, help="shortest face_valid run that becomes an episode")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="longer runs are split into pieces of at most this")
    ap.add_argument("--env-state", default="audio", choices=ENV_STATE_CHOICES,
                    help="what to duplicate into observation.environment_state (ACT/Diffusion need it without cameras)")
    ap.add_argument("--max-stretch", type=float, default=1.1,
                    help="drop runs the speed limit stretched by more than this factor (0 = keep all)")
    ap.add_argument("--force", action="store_true", help="replace an existing dataset at --out")
    ap.add_argument("--validate", action="store_true", help="load the result with lerobot in the validation venv")
    ap.add_argument("--lerobot-python", default=None, help="interpreter of the lerobot venv (default: .venv-lerobot)")
    ap.add_argument("--push", default=None, metavar="REPO_ID", help="upload to the Hub as a dataset (implies --validate)")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    exclude = [x for x in a.exclude.split(",") if x]
    print(f"export robot={a.robot} clips={a.clips} out={a.out} fps={a.fps:g} mode={a.mode} env_state={a.env_state}")
    summary = export(a.robot, a.clips, a.out, fps=a.fps, mode=a.mode, exclude=exclude, min_seconds=a.min_seconds,
                     max_seconds=a.max_seconds, env_state=a.env_state, force=a.force,
                     max_stretch=(a.max_stretch if a.max_stretch > 0 else None))
    print(f"wrote {a.out}: {summary['total_episodes']} episodes, {summary['total_frames']} frames "
          f"({summary['total_frames'] / summary['fps'] / 60:.1f} min), {summary['total_tasks']} tasks, "
          f"max time-stretch {summary['stretch_max']:.4f}")
    for t in summary["tasks"]:
        print("  task:", t)
    for s in summary["skipped"]:
        print("  skipped:", s)

    if a.validate or a.push:
        py = a.lerobot_python or default_lerobot_python(ROOT)
        print(f"validating with lerobot: {py}")
        ok, log, vsum = validate_with_lerobot(a.out, repo_id=a.push or f"animacy/{os.path.basename(a.out)}", python_exe=py)
        if not ok:
            print("VALIDATION FAILED")
            print(log[-4000:])
            return 1
        print("VALIDATION OK")
        print(json.dumps(vsum, indent=1))
        if a.push:
            url = push_to_hub(a.out, a.push, private=a.private)
            print("pushed", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
