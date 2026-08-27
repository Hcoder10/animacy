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
    ap.add_argument("--robots", default="", help="comma list of extra robots to open via ?robots= (e.g. so101)")
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
        q = "?autoplay=0" if "?" not in a.url else "&autoplay=0"
        if a.robots:
            q += f"&robots={a.robots}"
        page.goto(a.url + q, wait_until="domcontentloaded", timeout=a.timeout)
        page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=a.timeout)
        print(f"ready in {time.time() - t0:.1f}s")
        page.wait_for_timeout(1000)
        info = page.evaluate("({lamp: window.animacy.robotInfo('lamp'), reachy: window.animacy.robotInfo('reachy_mini'), native: window.animacy.clips.native.length, canonical: window.animacy.clips.canonical.map(c => c.id), errors: window.animacy.errors, status: document.getElementById('status').textContent, loaded: window.animacy.loadedRobots ? window.animacy.loadedRobots() : null, manifest: window.animacy.manifestRobots ? window.animacy.manifestRobots() : null})")
        print(f"robots loaded {info['loaded']} of manifest {info['manifest']}")
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
        for extra in [x for x in a.robots.split(",") if x]:
            r = page.evaluate(f"window.animacy.robotInfo('{extra}')")
            if not r or r["standin"] or r["missingJoints"]:
                failures.append(f"extra robot {extra} did not load cleanly on Pages: {r}")
            else:
                print(f"{extra}: {r['urdfUrl']} OK, joints {r['urdfJoints']}")
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

        # talk mode over Pages: synthetic 24 kHz voice → features → every backend the deployed bundle offers
        has_talk = page.evaluate("typeof window.animacy.sayAudio === 'function'")
        if has_talk:
            page.evaluate("window.animacy.setSource('talk')")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'talk'", timeout=30_000)
            avail = page.evaluate("window.animacy.backends.available()")
            deployed = page.evaluate("fetch('models/model.json', {cache: 'no-cache'}).then(r => r.ok ? r.json() : null).then(m => m ? {archs: m.archs || null, default_arch: m.default_arch || null, default_backend: m.default_backend} : null)")
            print(f"talk on Pages: backends {avail}; deployed model.json: {deployed}")
            voice_js = """(() => { const sr = 24000, n = Math.round(2.5 * sr), out = new Float32Array(n);
                for (let i = 0; i < n; i++) { const t = i / sr; const env = (t % 0.45) < 0.25 ? Math.sin(Math.PI * ((t % 0.45) / 0.25)) : 0;
                  let v = 0; for (let k = 1; k < 6; k++) v += Math.sin(2 * Math.PI * 130 * k * t) / k; out[i] = 0.5 * env * v + 0.01 * (Math.random() - 0.5); }
                return Array.from(out); })()"""
            rest = {n: {j["name"]: j["rest"] for j in page.evaluate(f"window.animacy.robots['{n}'].profile.joints")} for n in ("lamp", "reachy_mini")}
            for backend in avail:
                t_b = time.time()
                info = page.evaluate(f"(async () => {{ const a = {voice_js}; return await window.animacy.sayAudio(a, 24000, '{backend}'); }})()")
                wall = time.time() - t_b
                for _ in range(60):
                    page.wait_for_timeout(100)
                    if page.evaluate("window.animacy.talkInfo().time") > 0.6:
                        break
                ti = page.evaluate("window.animacy.talkInfo()")
                dev = {}
                for n in ("lamp", "reachy_mini"):
                    v = joints(n)
                    dev[n] = max(abs(v[k] - rest[n][k]) for k in rest[n])
                print(f"  talk/{backend}: used={info['backend']}{('/' + info['arch']) if info.get('arch') else ''} frames={info['frames']} motion {info['motionMs']:.0f} ms (wall {wall:.1f} s incl. downloads) "
                      f"clock {ti['time']:.2f}/{ti['duration']:.2f} playing={ti['playing']} lamp |d| {dev['lamp']:.1f} reachy |d| {dev['reachy_mini']:.1f}")
                if info["backend"] != backend:
                    failures.append(f"talk/{backend} on Pages fell back to {info['backend']}")
                if info["frames"] < 70 or not ti["playing"] or ti["time"] < 0.3:
                    failures.append(f"talk/{backend} on Pages did not play ({info['frames']} frames, t={ti['time']})")
                if dev["lamp"] < 0.5 or dev["reachy_mini"] < 0.5:
                    failures.append(f"talk/{backend} on Pages did not move both robots")
            page.screenshot(path=os.path.join(SHOTS, "pages_04_talk.png"))
        else:
            print("talk mode not deployed yet (no animacy.sayAudio on the page)")
            failures.append("talk mode not on Pages")
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
