"""Poll the physical Reachy Mini's daemon and print what it reports back.

    python scripts/video/broll_daemon_poll.py --url http://192.168.1.60:8000

Read-only: it GETs `/api/state/present_head_pose`,
`/api/state/present_antenna_joint_positions` and `/api/state/present_body_yaw`
while whatever is driving the robot keeps driving it (the ambient loop,
scripts/reachy_ambient.py). It commands nothing.

The ambient loop plays a clip, then rests, so a fixed window often lands on the
rest. This polls for `--collect` seconds and prints the busiest `--seconds` of
that read-out, and says so on the first line. Every printed row is a real
sample at its real timestamp; nothing is interpolated or reordered.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEG = 180.0 / math.pi
KEYS = ("yaw", "pitch", "roll")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.1.60:8000")
    ap.add_argument("--collect", type=float, default=45.0, help="seconds to poll")
    ap.add_argument("--seconds", type=float, default=5.0, help="seconds of it to print")
    ap.add_argument("--hz", type=float, default=6.0)
    a = ap.parse_args()

    s = requests.Session()

    def get(path: str):
        r = s.get(a.url + path, timeout=2.0)
        r.raise_for_status()
        return r.json()

    rows: list[tuple[float, dict, list, float]] = []
    t0 = time.time()
    errs = 0
    while time.time() - t0 < a.collect:
        tick = time.time()
        try:
            hp = get("/api/state/present_head_pose")
            ant = get("/api/state/present_antenna_joint_positions")
            by = get("/api/state/present_body_yaw")
            by = by if isinstance(by, (int, float)) else list(by.values())[0]
            rows.append((tick - t0, hp, list(ant), float(by)))
        except Exception:  # noqa: BLE001
            errs += 1
        time.sleep(max(0.0, 1.0 / a.hz - (time.time() - tick)))
    if not rows:
        print(f"no reads from {a.url} ({errs} errors)")
        return 1

    # busiest contiguous window: the most total head movement between samples
    width = max(2, int(round(a.seconds * a.hz)))
    step = [0.0] + [sum(abs(rows[i][1][k] - rows[i - 1][1][k]) * DEG for k in KEYS)
                    for i in range(1, len(rows))]
    best, best_sum = 0, -1.0
    for i in range(0, max(1, len(rows) - width + 1)):
        tot = sum(step[i:i + width])
        if tot > best_sum:
            best, best_sum = i, tot
    win = rows[best:best + width]

    print(f"GET {a.url}/api/state/present_*   read-only, {a.hz:g} Hz, degrees")
    print(f"the busiest {win[-1][0] - win[0][0]:.1f} s of a {rows[-1][0]:.0f} s read-out "
          f"({len(rows)} samples, {errs} errors) while the ambient loop drove the robot")
    print(f"{'t':>6}  {'head_yaw':>9}{'head_pitch':>11}{'head_roll':>10}"
          f"{'ant_L':>8}{'ant_R':>8}{'body_yaw':>9}   {'head x/y/z mm':>17}")
    print("-" * 86)
    t_off = win[0][0]
    for t, hp, ant, by in win:
        print(f"{t - t_off:6.2f}  {hp['yaw'] * DEG:9.1f}{hp['pitch'] * DEG:11.1f}"
              f"{hp['roll'] * DEG:10.1f}{ant[0] * DEG:8.1f}{ant[1] * DEG:8.1f}"
              f"{by * DEG:9.1f}   {hp['x'] * 1000:5.1f} {hp['y'] * 1000:5.1f} {hp['z'] * 1000:5.1f}")
    print("-" * 86)
    rng = {k: max(r[1][k] for r in win) * DEG - min(r[1][k] for r in win) * DEG for k in KEYS}
    print("head travel over this window: "
          + "  ".join(f"{k} {rng[k]:.1f} deg" for k in KEYS)
          + f"   ({len(rows)} reads, {errs} errors overall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
