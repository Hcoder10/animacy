"""Headless verification of the animacy web viewer.

    python web/dev/screenshot.py [--port 8000] [--keep] [--skip-webcam]

Starts `python -m http.server` from the repo root, opens http://127.0.0.1:<port>/web/
in headless Chromium (Playwright), waits for both robots, drives the page through
the public `window.animacy` test API and writes screenshots to web/dev/shots/.

Asserts:
  * zero console errors / page errors (warnings are allowed and printed)
  * both robots loaded (reports whether they came from the real URDF or a stand-in)
  * joint values change during native `nod` / `headshake` playback on the lamp
  * a calibration clip moves BOTH robots with the documented signs
  * puppet mode + A/B viewport work
  * webcam mode initialises without throwing (fake camera; no face expected)
Exit code 1 on any failure. Run with --keep to leave the http server running.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SHOTS = os.path.join(HERE, "shots")


def wait_port(port: int, timeout: float = 15.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"http server did not come up on {port}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-webcam", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--query", default="", help="extra query string, e.g. standin=1")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    sys.path.insert(0, HERE)
    import build_manifest  # noqa: E402

    build_manifest.main()
    os.makedirs(SHOTS, exist_ok=True)
    port = a.port or free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures: list[str] = []
    console_errors: list[str] = []
    warnings: list[str] = []
    try:
        wait_port(port)
        url = f"http://127.0.0.1:{port}/web/?autoplay=0" + (f"&{a.query}" if a.query else "")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=not a.headed,
                args=[
                    "--use-fake-device-for-media-capture",
                    "--use-fake-ui-for-media-stream",
                    "--ignore-gpu-blocklist",
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera", "microphone"])
            page = ctx.new_page()

            # MediaPipe's wasm runtime routes its own INFO lines through console.error; not ours.
            benign = ("INFO: Created TensorFlow Lite XNNPACK delegate",)

            def on_console(msg):
                if msg.type == "error":
                    if any(b in msg.text for b in benign):
                        warnings.append(msg.text)
                    else:
                        console_errors.append(msg.text)
                elif msg.type == "warning":
                    warnings.append(msg.text)

            page.on("console", on_console)
            page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

            def shot(name: str):
                path = os.path.join(SHOTS, f"{name}.png")
                page.screenshot(path=path)
                print(f"  shot {os.path.relpath(path, ROOT)}")

            def ev(js: str):
                return page.evaluate(js)

            def joints(robot: str):
                return ev(f"window.animacy.getJointValues('{robot}')")

            def changed(v0, v1, tol=0.5):
                return any(abs(v1[k] - v0[k]) > tol for k in v0)

            print(f"open {url}")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            page.wait_for_timeout(800)
            info = ev("({lamp: window.animacy.robotInfo('lamp'), reachy: window.animacy.robotInfo('reachy_mini'), errors: window.animacy.errors})")
            print("robots:", json.dumps(info, indent=1))
            for n in ("lamp", "reachy"):
                if not info[n]:
                    failures.append(f"{n} did not load")
            shot("01_initial")

            # ---- native clips on the lamp -------------------------------------
            if info["lamp"]:
                for clip, seek_frac in (("nod", 0.35), ("headshake", 0.55)):
                    ev(f"(async () => {{ await window.animacy.setSource('native'); await window.animacy.setClip('lamp/{clip}'); }})()")
                    page.wait_for_function(f"window.animacy.sourceInfo().clip === 'lamp/{clip}'", timeout=20_000)
                    ev("window.animacy.pause(); window.animacy.seek(0)")
                    page.wait_for_timeout(120)
                    v0 = joints("lamp")
                    dur = ev("window.animacy.sourceInfo().duration")
                    # play from the start and record the largest excursion over a full pass
                    ev("window.animacy.seek(0); window.animacy.play()")
                    best, best_t = 0.0, 0.0
                    t_end = time.time() + min(dur, 4.0) + 0.3
                    while time.time() < t_end:
                        v = joints("lamp")
                        dev = max(abs(v[k] - v0[k]) for k in v0)
                        if dev > best:
                            best, best_t = dev, ev("window.animacy.sourceInfo().time")
                        page.wait_for_timeout(60)
                    print(f"native {clip}: dur={dur:.2f}s rest={ {k: round(v, 1) for k, v in v0.items()} } max excursion {best:.1f} deg at t={best_t:.2f}s")
                    if best < 3.0:
                        failures.append(f"lamp joints did not change during native '{clip}' (max excursion {best:.2f} deg)")
                    ev(f"window.animacy.seek({dur * seek_frac}); window.animacy.play()")
                    page.wait_for_timeout(150)
                    shot(f"0{2 if clip == 'nod' else 3}_native_{clip}_mid")

            # ---- calibration clip on both robots ------------------------------
            ev("(async () => { await window.animacy.setSource('canonical'); await window.animacy.setClip('synth/cal_look_left_right'); })()")
            page.wait_for_function("window.animacy.sourceInfo().clip === 'synth/cal_look_left_right'", timeout=20_000)
            ev("window.animacy.pause(); window.animacy.seek(0)")
            page.wait_for_timeout(120)
            l0, r0 = joints("lamp"), joints("reachy_mini")
            ev("window.animacy.seek(0.55); window.animacy.play()")  # peak head_yaw=+30 at t=1.0 s
            page.wait_for_timeout(520)
            l1, r1 = joints("lamp"), joints("reachy_mini")
            ch = ev("window.animacy.getChannels()")
            print(f"cal look-left: head_yaw={ch['head_yaw']:.1f} lamp wrist_roll {l0['wrist_roll']:.1f}->{l1['wrist_roll']:.1f} base_yaw {l0['base_yaw']:.1f}->{l1['base_yaw']:.1f} | reachy head_yaw {r0['head_yaw']:.1f}->{r1['head_yaw']:.1f} body_yaw {r0['body_yaw']:.1f}->{r1['body_yaw']:.1f}")
            if ch["head_yaw"] < 15:
                failures.append(f"expected head_yaw near +30 at t≈1 s, got {ch['head_yaw']}")
            if not (l1["wrist_roll"] > l0["wrist_roll"] + 5):
                failures.append("lamp wrist_roll did not go POSITIVE on look-left (default mapping gain +1)")
            if not (r1["head_yaw"] > r0["head_yaw"] + 5):
                failures.append("reachy head_yaw did not go POSITIVE on look-left")
            shot("04_cal_look_left_both")
            # front view, frozen at the peak: the robots' LEFT must be on the viewer's RIGHT
            ev("window.animacy.pause(); window.animacy.seek(1.0); window.animacy.setView('front')")
            page.wait_for_timeout(600)
            shot("04b_cal_look_left_FRONT_view")
            ev("(async () => { await window.animacy.setClip('synth/cal_look_up_down'); window.animacy.pause(); window.animacy.seek(1.0); })()")
            page.wait_for_timeout(700)
            shot("04c_cal_look_up_FRONT_view")
            ev("window.animacy.setView('iso'); window.animacy.play()")

            # brows → antennas / wrist_pitch
            ev("(async () => { await window.animacy.setClip('synth/cal_brows'); })()")
            page.wait_for_function("window.animacy.sourceInfo().clip === 'synth/cal_brows'", timeout=20_000)
            ev("window.animacy.seek(0.2); window.animacy.play()")
            page.wait_for_timeout(350)
            rb = joints("reachy_mini")
            chb = ev("window.animacy.getChannels()")
            print(f"cal brows: brow_l={chb['brow_l']:.2f} antenna_left={rb['antenna_left']:.1f} antenna_right={rb['antenna_right']:.1f}")
            if rb["antenna_left"] < 10:
                failures.append("reachy antenna_left did not rise on brow raise")
            shot("05_cal_brows_both")

            # ---- puppet mode ---------------------------------------------------
            ev("(async () => { window.animacy.setMode('puppet'); await window.animacy.setClip('synth/cal_puppet_wave'); })()")
            page.wait_for_function("window.animacy.sourceInfo().clip === 'synth/cal_puppet_wave'", timeout=20_000)
            ev("window.animacy.seek(0.8); window.animacy.play()")
            page.wait_for_timeout(900)
            lp = joints("lamp")
            print(f"puppet wave: lamp={ {k: round(v, 1) for k, v in lp.items()} }")
            if abs(lp["elbow_pitch"] - 27.6) < 1 and abs(lp["base_pitch"] - 28.9) < 1:
                failures.append("puppet mode did not move the lamp")
            shot("06_puppet_wave")
            ev("window.animacy.setMode('default')")

            # ---- A/B viewport --------------------------------------------------
            ev("(async () => { await window.animacy.setClip('synth/cal_nod'); await window.animacy.setAb(true); await window.animacy.setAbClip('lamp/nod'); })()")
            page.wait_for_function("window.animacy.ab.on && window.animacy.ab.viewer && window.animacy.ab.source", timeout=60_000)
            ev("window.animacy.seek(0.1); window.animacy.play()")
            ab_best = 0.0
            t_end = time.time() + 2.5
            while time.time() < t_end:  # the vendor nod dips at ~0.7-1.3 s of its 1.95 s loop
                v = ev("window.animacy.ab.viewer ? window.animacy.ab.viewer.values : null") or {}
                ab_best = max(ab_best, abs(v.get("elbow_pitch", 27.1) - 27.1))
                page.wait_for_timeout(60)
            print(f"A/B lamp (B = vendor nod): max elbow excursion {ab_best:.1f} deg")
            if ab_best < 3.0:
                failures.append("A/B lamp is not playing the vendor clip")
            ev("window.animacy.pause(); window.animacy.seek(0.1); window.animacy.ab.source.seek(0.75)")
            page.wait_for_timeout(300)
            shot("07_ab_nod_vs_vendor_nod")
            ev("window.animacy.play(); window.animacy.setAb(false)")

            # ---- every captured clip in web/clips must drive BOTH robots ---------
            captured = ev("window.animacy.clips.canonical.filter(c => c.group === 'captured').map(c => c.id)")
            rest_l = {j["name"]: j["rest"] for j in ev("window.animacy.robots.lamp.profile.joints")}
            rest_r = {j["name"]: j["rest"] for j in ev("window.animacy.robots.reachy_mini.profile.joints")}
            for k, cid in enumerate(captured):
                ev(f"(async () => {{ await window.animacy.setClip('{cid}'); }})()")
                page.wait_for_function(f"window.animacy.sourceInfo().clip === '{cid}'", timeout=120_000)
                dur = ev("window.animacy.sourceInfo().duration")
                ev(f"window.animacy.seek({min(20.0, dur * 0.3)}); window.animacy.play()")
                best_l, best_r, valid = 0.0, 0.0, 0
                t_end = time.time() + 3.0
                while time.time() < t_end:
                    lc, rc = joints("lamp"), joints("reachy_mini")
                    chc = ev("window.animacy.getChannels()")
                    best_l = max(best_l, max(abs(lc[j] - rest_l[j]) for j in rest_l))
                    best_r = max(best_r, max(abs(rc[j] - rest_r[j]) for j in rest_r))
                    valid += int(chc["face_valid"] == 1)
                    page.wait_for_timeout(100)
                print(f"captured {cid}: dur={dur:.1f}s, over 3 s from t={min(20.0, dur * 0.3):.0f}: lamp max |joint-rest| {best_l:.1f}, reachy {best_r:.1f} (deg/mm), face_valid samples {valid}")
                if best_l < 1.0 or best_r < 1.0:
                    failures.append(f"captured clip {cid} did not move both robots (lamp {best_l:.2f}, reachy {best_r:.2f})")
                if k == 0:
                    shot("09_captured_clip")
            if not captured:
                print("no captured clips in web/clips (skipped)")

            # ---- Model tab: must degrade gracefully with no web/models/*.onnx ------
            ev("window.animacy.setSource('model')")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'model'", timeout=20_000)
            page.wait_for_timeout(600)
            model_status = ev("({status: document.getElementById('status').textContent, cls: document.getElementById('status').className, models: (window.animacy.source && window.animacy.source.models) || []})")
            print(f"model tab: {model_status['status']!r} ({len(model_status['models'])} model file(s) in manifest)")
            if "err" in model_status["cls"]:
                failures.append(f"model tab reported an error: {model_status['status']}")
            shot("10_model_tab")
            ev("window.animacy.setSource('canonical')")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'canonical'", timeout=20_000)

            # ---- FPS -----------------------------------------------------------
            ev("(async () => { await window.animacy.setClip('synth/cal_talk'); })()")
            page.wait_for_timeout(2500)
            fps = ev("window.animacy.fps")
            print(f"render fps (headless swiftshader, not representative of a laptop GPU): {fps:.1f}")

            # ---- webcam mode: must initialise without throwing ---------------
            if not a.skip_webcam:
                ev("window.animacy.setSource('webcam').catch(e => { window.__wcErr = String(e && e.message || e); })")
                page.wait_for_function(
                    "window.__wcErr || (window.animacy.webcam && window.animacy.webcam.running) || (window.animacy.errors.some(e => e.startsWith('source webcam')))",
                    timeout=180_000,
                )
                page.wait_for_timeout(2500)
                wc = ev("({err: window.__wcErr || null, running: !!(window.animacy.webcam && window.animacy.webcam.running), fps: window.animacy.webcam && window.animacy.webcam.fps, face: window.animacy.webcam && window.animacy.webcam.hasFace, status: document.getElementById('webcam-status').textContent, errors: window.animacy.errors})")
                print("webcam:", json.dumps(wc))
                if wc["err"] or not wc["running"]:
                    failures.append(f"webcam mode failed to initialise: {wc['err'] or wc['errors']}")
                shot("08_webcam_mode")
                ev("window.animacy.setSource('native')")

            app_errors = ev("window.animacy.errors")
            if app_errors:
                failures.append(f"app reported errors: {app_errors}")
            browser.close()
    finally:
        if not a.keep:
            srv.terminate()

    print()
    if warnings:
        print(f"{len(warnings)} console warning(s):")
        for w in warnings[:10]:
            print("  -", w[:300])
    if console_errors:
        print(f"{len(console_errors)} console error(s):")
        for e in console_errors:
            print("  -", e[:500])
        failures.append(f"{len(console_errors)} console error(s)")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("OK — no console errors, all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
