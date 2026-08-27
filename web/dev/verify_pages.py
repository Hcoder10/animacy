"""Verify the LIVE GitHub Pages deployment of the viewer.

    python web/dev/verify_pages.py [--url https://hcoder10.github.io/animacy/web/]

Opens the deployed page in headless Chromium, waits for both robots, checks the
URDFs/meshes/clips/manifest actually came over Pages (relative paths,
octet-stream STLs), plays a native and a canonical clip, requires zero console
errors and writes web/dev/shots/pages_*.png. Exit 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "shots")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://hcoder10.github.io/animacy/web/")
    ap.add_argument("--timeout", type=int, default=180_000)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS, exist_ok=True)
    failures, console_errors, warnings, requests = [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
        benign = ("INFO: Created TensorFlow Lite XNNPACK delegate",)
        page.on("console", lambda m: (console_errors if m.type == "error" and not any(b in m.text for b in benign) else warnings).append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        def on_response(r):
            try:
                requests.append({"url": r.url, "status": r.status, "type": r.headers.get("content-type", "")})
            except Exception:  # noqa: BLE001
                pass

        page.on("response", on_response)
        t0 = time.time()
        print(f"open {a.url}")
        page.goto(a.url + ("?autoplay=0" if "?" not in a.url else "&autoplay=0"), wait_until="domcontentloaded", timeout=a.timeout)
        page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=a.timeout)
        print(f"ready in {time.time() - t0:.1f}s")
        page.wait_for_timeout(1000)
        info = page.evaluate("({lamp: window.animacy.robotInfo('lamp'), reachy: window.animacy.robotInfo('reachy_mini'), native: window.animacy.clips.native.length, canonical: window.animacy.clips.canonical.map(c => c.id), errors: window.animacy.errors, status: document.getElementById('status').textContent})")
        for n in ("lamp", "reachy"):
            r = info[n]
            if not r:
                failures.append(f"{n} did not load")
            elif r["standin"]:
                failures.append(f"{n} fell back to the stand-in URDF ({r['urdfUrl']})")
            elif r["missingJoints"]:
                failures.append(f"{n} missing joints in URDF: {r['missingJoints']}")
            else:
                print(f"{n}: {r['urdfUrl']} OK, joints {r['urdfJoints']}")
        print(f"native clips listed: {info['native']}, canonical: {info['canonical']}")
        if info["native"] < 40:
            failures.append(f"expected >= 47 native clips (31 lamp + 16 reachy), got {info['native']}")
        if not any(c.startswith("clip/") for c in info["canonical"]):
            failures.append("no captured clips listed (manifest.json not served or empty)")
        # what actually came over the wire
        by_ext = {}
        bad = []
        for r in requests:
            u = r["url"].split("?")[0]
            ext = u.rsplit(".", 1)[-1].lower() if "." in u.rsplit("/", 1)[-1] else "(none)"
            by_ext.setdefault(ext, []).append(r)
            if r["status"] >= 400:
                bad.append(f"{r['status']} {u}")
        for ext in ("urdf", "stl", "json", "csv", "js"):
            rs = by_ext.get(ext, [])
            if rs:
                print(f"  {ext}: {len(rs)} responses, e.g. {rs[0]['status']} {rs[0]['type'] or '(no content-type)'}")
        if bad:
            failures.append(f"{len(bad)} failed request(s): {bad[:8]}")
        if not by_ext.get("stl"):
            failures.append("no .stl meshes were fetched")
        page.screenshot(path=os.path.join(SHOTS, "pages_01_initial.png"))

        # play a native clip and a canonical clip: joints must move
        def joints(robot):
            return page.evaluate(f"window.animacy.getJointValues('{robot}')")

        page.evaluate("(async () => { await window.animacy.setSource('native'); await window.animacy.setClip('lamp/headshake'); window.animacy.play(); })()")
        page.wait_for_function("window.animacy.sourceInfo().clip === 'lamp/headshake'", timeout=60_000)
        v0 = joints("lamp")
        best = 0.0
        for _ in range(25):
            page.wait_for_timeout(100)
            v = joints("lamp")
            best = max(best, max(abs(v[k] - v0[k]) for k in v0))
        print(f"native headshake on Pages: max excursion {best:.1f} deg")
        if best < 3:
            failures.append("native clip did not move the lamp on Pages")
        page.screenshot(path=os.path.join(SHOTS, "pages_02_native_headshake.png"))
        page.evaluate("(async () => { await window.animacy.setSource('canonical'); await window.animacy.setClip('synth/cal_look_left_right'); window.animacy.seek(0.8); window.animacy.play(); })()")
        page.wait_for_function("window.animacy.sourceInfo().clip === 'synth/cal_look_left_right'", timeout=60_000)
        page.wait_for_timeout(700)
        r = joints("reachy_mini")
        print(f"canonical look-left on Pages: reachy head_yaw {r['head_yaw']:.1f}")
        if abs(r["head_yaw"]) < 5:
            failures.append("canonical clip did not move the reachy on Pages")
        page.screenshot(path=os.path.join(SHOTS, "pages_03_canonical.png"))
        app_errors = page.evaluate("window.animacy.errors")
        if app_errors:
            failures.append(f"app errors: {app_errors}")
        browser.close()

    if console_errors:
        failures.append(f"{len(console_errors)} console error(s): {console_errors[:5]}")
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("OK — live deployment serves both robots, meshes, clips; no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
