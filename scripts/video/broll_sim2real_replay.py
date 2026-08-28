"""Print commanded vs measured degrees from a logged sim-to-real run.

    python scripts/video/broll_sim2real_replay.py data/sim2real/reachy_20260826_214727.json

The log is what `scripts/reachy_sim2real.py` recorded off the physical Reachy
Mini's daemon (`present_head_pose`, `present_antenna_joint_positions`,
`present_body_yaw`) while it drove the robot. This only formats it; it issues
no motion and invents no numbers.

Frames: the log stores `commanded_peak.head_pitch` in the CANONICAL frame
(+pitch = up) and the daemon reports pitch in its own frame (+pitch = down), so
the commanded pitch is multiplied by the run's own `pitch_sign` before it is
compared — exactly as docs/evidence/reachy_sim2real_20260826.md does.
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT = os.path.join(ROOT, "data", "sim2real", "reachy_20260826_214727.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEG = 180.0 / math.pi
# (column header, commanded key, measured getter, group)
AXES = [
    ("head_yaw", "head_yaw", lambda m: m["head_pose_rad_m"]["yaw"] * DEG, "head"),
    ("head_pitch", "head_pitch", lambda m: m["head_pose_rad_m"]["pitch"] * DEG, "head"),
    ("head_roll", "head_roll", lambda m: m["head_pose_rad_m"]["roll"] * DEG, "head"),
    ("body_yaw", "body_yaw", lambda m: m["body_yaw_rad"] * DEG, "head"),
    ("antenna_l", "antenna_left", lambda m: m["antennas_rad"][0] * DEG, "antenna"),
    ("antenna_r", "antenna_right", lambda m: m["antennas_rad"][1] * DEG, "antenna"),
]


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as fh:
        log = json.load(fh)
    pitch_sign = float(log.get("pitch_sign", 1.0))
    segs = [s for s in log["segments"] if s["label"] != "center"]

    print(f"host {log['host']}   daemon HTTP API @ 30 Hz   started {log['started'][:19]}")
    print(f"{len(log['segments'])} segments logged; the {len(segs)} commanded moves below. Degrees, daemon frame.")
    print()
    print(f"{'segment':<22}{'axis':<12}{'commanded':>11}{'measured':>11}{'error':>9}")
    print("-" * 65)
    worst = {"head": 0.0, "antenna": 0.0}
    for s in segs:
        first = True
        for name, ckey, get, group in AXES:
            cmd = float(s["commanded_peak"].get(ckey, 0.0))
            if ckey == "head_pitch":
                cmd *= pitch_sign          # canonical +up -> what was actually sent
            meas = float(get(s["measured_peak"]))
            if abs(cmd) < 0.5 and abs(meas) < 2.0:
                continue                   # axis not exercised by this segment
            err = meas - cmd
            worst[group] = max(worst[group], abs(err))
            print(f"{(s['label'] if first else ''):<22}{name:<12}"
                  f"{cmd:>11.1f}{meas:>11.1f}{err:>+9.1f}")
            first = False
    print("-" * 65)
    print(f"head + body axes: every commanded axis read back within {worst['head']:.1f} deg")
    print(f"antennas: worst {worst['antenna']:.1f} deg (the right antenna under-reports at the brow peak)")
    print(f"source: {os.path.relpath(path, ROOT).replace(os.sep, '/')}  "
          f"evidence: docs/evidence/reachy_sim2real_20260826.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
