"""High-frame-rate stability check of the retargeter on the GPU (headed Chromium).

    python web/dev/jitter.py [--seconds 3] [--clip synth/cal_look_left_right]

The fitted ROBOT.md profiles use spring trackers (hz 2-4) and idle sway; the
viewer steps the retargeter once per animation frame, so at 240 Hz dt is ~4 ms.
This samples both robots' joint values every frame for a few seconds while a
smooth calibration clip plays and reports the render rate, the largest
frame-to-frame step and the largest second difference (jerk) per joint. A
stable tracker on a smooth input gives small, bounded second differences; an
unstable one shows growing / alternating steps.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screenshot import ROOT, free_port, wait_port  # noqa: E402

TRACE_JS = """
(async (seconds) => {
  const names = ['lamp', 'reachy_mini'];
  const rows = [];
  const t0 = performance.now();
  await new Promise((resolve) => {
    const tick = (now) => {
      rows.push({ t: (now - t0) / 1000, lamp: window.animacy.getJointValues('lamp'), reachy_mini: window.animacy.getJointValues('reachy_mini') });
      if (now - t0 < seconds * 1000) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
  return rows;
})
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--clip", default="synth/cal_look_left_right")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = 0
    try:
        wait_port(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=a.headless, args=["--ignore-gpu-blocklist"])
            page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
            page.goto(f"http://127.0.0.1:{port}/web/", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            page.evaluate(f"(async () => {{ await window.animacy.setSource('canonical'); await window.animacy.setClip('{a.clip}'); window.animacy.play(); }})()")
            page.wait_for_function(f"window.animacy.sourceInfo().clip === '{a.clip}'", timeout=30_000)
            page.wait_for_timeout(1500)
            rows = page.evaluate(TRACE_JS, a.seconds)
            browser.close()
    finally:
        srv.terminate()
    n = len(rows)
    fps = (n - 1) / max(1e-6, rows[-1]["t"] - rows[0]["t"])
    print(f"{n} frames in {rows[-1]['t']:.2f} s → {fps:.0f} fps; clip {a.clip}")
    worst = 0.0
    for robot in ("lamp", "reachy_mini"):
        for j in rows[0][robot]:
            v = [r[robot][j] for r in rows]
            d1 = [abs(v[i + 1] - v[i]) for i in range(n - 1)]
            d2 = [abs(v[i + 2] - 2 * v[i + 1] + v[i]) for i in range(n - 2)]
            rng = max(v) - min(v)
            if rng < 0.05:
                continue
            print(f"  {robot:12s} {j:14s} range {rng:6.1f}  max step {max(d1):6.3f}  max jerk {max(d2):6.3f}  (per frame)")
            worst = max(worst, max(d2) / max(rng, 1e-6))
    print(f"worst second-difference / range = {worst:.4f}  ({'stable' if worst < 0.05 else 'SUSPICIOUS'})")
    if worst >= 0.05:
        rc = 1
    print("OK" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
