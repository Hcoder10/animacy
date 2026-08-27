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
  * every captured clip in web/clips drives both robots
  * Talk tab: a synthetic 24 kHz voice goes through features → each available
    backend (model / retrieval / envelope) → both robots, clocked to WebAudio
    (--with-tts additionally runs Kokoro TTS in the page)
  * Listen tab starts on the fake microphone
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

# Windows consoles default to cp1252; the page's status strings are UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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
    ap.add_argument("--with-tts", action="store_true", help="also run Kokoro TTS in the page (downloads ~90 MB once)")
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
            # ...and kokoro-js' bundled onnxruntime logs its EP-assignment warnings the same way.
            benign = ("INFO: Created TensorFlow Lite XNNPACK delegate", "[W:onnxruntime", "Unable to determine content-length")

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
            fwd0 = {n: ev(f"window.animacy.linkForward('{n}', 'head')") for n in ("lamp", "reachy_mini")}
            ev("window.animacy.seek(0.55); window.animacy.play()")  # peak head_yaw=+30 at t=1.0 s
            page.wait_for_timeout(520)
            l1, r1 = joints("lamp"), joints("reachy_mini")
            ch = ev("window.animacy.getChannels()")
            fwd1 = {n: ev(f"window.animacy.linkForward('{n}', 'head')") for n in ("lamp", "reachy_mini")}
            print(f"cal look-left: head_yaw={ch['head_yaw']:.1f} lamp wrist_roll {l0['wrist_roll']:.1f}->{l1['wrist_roll']:.1f} base_yaw {l0['base_yaw']:.1f}->{l1['base_yaw']:.1f} | reachy head_yaw {r0['head_yaw']:.1f}->{r1['head_yaw']:.1f} body_yaw {r0['body_yaw']:.1f}->{r1['body_yaw']:.1f}")
            if ch["head_yaw"] < 15:
                failures.append(f"expected head_yaw near +30 at t≈1 s, got {ch['head_yaw']}")
            # physical check, independent of ROBOT.md gain signs: each head's forward axis must swing
            # toward the robot's LEFT (three.js −z) when the human looks left
            for n in ("lamp", "reachy_mini"):
                if fwd0[n] and fwd1[n]:
                    dz = fwd1[n]["z"] - fwd0[n]["z"]
                    print(f"  {n} head forward z: {fwd0[n]['z']:+.3f} -> {fwd1[n]['z']:+.3f} ({'LEFT' if dz < 0 else 'RIGHT'})")
                    if dz > -0.05:
                        failures.append(f"{n}: head did not turn LEFT on canonical head_yaw=+30 (Δz={dz:+.3f}); check the yaw gains in robots/{n}/ROBOT.md vs the URDF axis")
                else:
                    warnings.append(f"{n}: no link named 'head' — left/right check skipped")
            shot("04_cal_look_left_both")
            # front view, frozen at the peak: the robots' LEFT must be on the viewer's RIGHT
            ev("window.animacy.pause(); window.animacy.seek(1.0); window.animacy.setView('front')")
            page.wait_for_timeout(600)
            shot("04b_cal_look_left_FRONT_view")
            ev("(async () => { await window.animacy.setClip('synth/cal_look_up_down'); window.animacy.pause(); window.animacy.seek(0); })()")
            page.wait_for_timeout(500)
            up0 = {n: ev(f"window.animacy.linkForward('{n}', 'head')") for n in ("lamp", "reachy_mini")}
            ev("window.animacy.seek(0.55); window.animacy.play()")  # peak head_pitch=+20 at t=1.0 s
            page.wait_for_timeout(520)
            up1 = {n: ev(f"window.animacy.linkForward('{n}', 'head')") for n in ("lamp", "reachy_mini")}
            for n in ("lamp", "reachy_mini"):
                if up0[n] and up1[n]:
                    dy = up1[n]["y"] - up0[n]["y"]
                    print(f"  look-up: {n} head forward y: {up0[n]['y']:+.3f} -> {up1[n]['y']:+.3f} ({'UP' if dy > 0 else 'DOWN'})")
                    if dy < 0.03:
                        failures.append(f"{n}: head did not tip UP on canonical head_pitch=+20 (Δy={dy:+.3f})")
            ev("window.animacy.pause(); window.animacy.seek(1.0)")
            page.wait_for_timeout(400)
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
                    valid += int(bool(chc) and chc["face_valid"] == 1)
                    page.wait_for_timeout(100)
                print(f"captured {cid}: dur={dur:.1f}s, over 3 s from t={min(20.0, dur * 0.3):.0f}: lamp max |joint-rest| {best_l:.1f}, reachy {best_r:.1f} (deg/mm), face_valid samples {valid}")
                if best_l < 1.0 or best_r < 1.0:
                    failures.append(f"captured clip {cid} did not move both robots (lamp {best_l:.2f}, reachy {best_r:.2f})")
                if k == 0:
                    shot("09_captured_clip")
            if not captured:
                print("no captured clips in web/clips (skipped)")

            # ---- Talk tab: synthetic voice → features → every available backend → both robots --
            # (TTS itself needs the ~90 MB Kokoro download: --with-tts runs it too)
            ev("window.animacy.setSource('talk')")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'talk'", timeout=20_000)
            page.wait_for_timeout(300)
            talk0 = ev("({status: document.getElementById('status').textContent, cls: document.getElementById('status').className, available: window.animacy.backends.available()})")
            print(f"talk tab: {talk0['status']!r}")
            if "err" in talk0["cls"]:
                failures.append(f"talk tab reported an error: {talk0['status']}")
            # a 2.5 s synthetic voice at 24 kHz (what Kokoro would hand back)
            voice_js = """(() => { const sr = 24000, n = Math.round(2.5 * sr), out = new Float32Array(n);
                for (let i = 0; i < n; i++) { const t = i / sr; const env = (t % 0.45) < 0.25 ? Math.sin(Math.PI * ((t % 0.45) / 0.25)) : 0;
                  let v = 0; for (let k = 1; k < 6; k++) v += Math.sin(2 * Math.PI * 130 * k * t) / k; out[i] = 0.5 * env * v + 0.01 * (Math.random() - 0.5); }
                return Array.from(out); })()"""
            for backend in talk0["available"]:
                t_b = time.time()
                info = ev(f"(async () => {{ const a = {voice_js}; return await window.animacy.sayAudio(a, 24000, '{backend}'); }})()")
                elapsed = time.time() - t_b
                # motion is clocked to the AudioContext, which headless Chromium can take a few
                # seconds to start the first time; the robots hold until the audio really runs
                for _ in range(60):
                    page.wait_for_timeout(100)
                    ti = ev("window.animacy.talkInfo()")
                    if ti["time"] > 0.6:
                        break
                page.wait_for_timeout(300)
                ti = ev("window.animacy.talkInfo()")
                lc, rc = joints("lamp"), joints("reachy_mini")
                best_l = max(abs(lc[j] - rest_l[j]) for j in rest_l)
                best_r = max(abs(rc[j] - rest_r[j]) for j in rest_r)
                print(f"talk/{backend}: used={info['backend']} frames={info['frames']} motion {info['motionMs']:.0f} ms (wall {elapsed:.1f} s incl. load) · audio clock t={ti['time']:.2f}/{ti['duration']:.2f} playing={ti['playing']} · lamp |Δ| {best_l:.1f} reachy |Δ| {best_r:.1f}")
                if info["backend"] != backend:
                    failures.append(f"talk backend '{backend}' fell back to '{info['backend']}'")
                if info["frames"] < 70:
                    failures.append(f"talk/{backend}: expected ~75 frames for 2.5 s, got {info['frames']}")
                if not ti["playing"] or ti["time"] < 0.3:
                    failures.append(f"talk/{backend}: audio clock did not advance (t={ti['time']}, playing={ti['playing']})")
                if best_l < 0.5 or best_r < 0.5:
                    failures.append(f"talk/{backend}: robots did not move (lamp {best_l:.2f}, reachy {best_r:.2f})")
                if backend == talk0["available"][0]:
                    shot("10_talk_tab")
            if a.with_tts:
                t_b = time.time()
                said = ev("(async () => await window.animacy.say('Hello from animacy.'))()")
                print(f"talk/TTS: {said} in {time.time() - t_b:.1f} s")
                if not said:
                    failures.append("TTS say() failed (see app errors)")
                page.wait_for_timeout(1500)
                shot("11_talk_tts")

            # ---- Listen tab: must start on the fake microphone without throwing --------
            ev("window.animacy.setSource('listen').catch(e => { window.__listenErr = String(e && e.message || e); })")
            page.wait_for_function("window.__listenErr || window.animacy.sourceInfo().kind === 'listen'", timeout=30_000)
            page.wait_for_timeout(800)
            li = ev("({err: window.__listenErr || null, kind: window.animacy.sourceInfo().kind, running: !!(window.animacy.source && window.animacy.source.running), stat: document.getElementById('listen-stat').textContent})")
            print(f"listen tab: {li}")
            if li["err"] or not li["running"]:
                failures.append(f"listen mode failed to initialise: {li}")
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
                # ---- record mode: a 1.5 s take on the fake camera + fake microphone --------
                if wc["running"]:
                    import base64
                    import io
                    import zipfile

                    ev("window.__recErr = null; window.animacy.record.start({subject: 'ci', slug: 'smoke', role: 'speaking', prompt: 'smoke test', seconds: 1.5}).catch(e => { window.__recErr = String(e && e.message || e); })")
                    page.wait_for_function("window.__recErr || (window.animacy.recorder && window.animacy.recorder.recording)", timeout=20_000)
                    page.wait_for_timeout(1500)
                    ev("window.animacy.record.stop().catch(e => { window.__recErr = String(e && e.message || e); })")
                    page.wait_for_function("window.__recErr || window.animacy.record.lastTake()", timeout=20_000)
                    rec_err = ev("window.__recErr")
                    take = ev("(() => { const t = window.animacy.record.lastTake(); if (!t) return null; const { motion, ...rest } = t; return { ...rest, n_channels: motion.channels.length, n_t: motion.data.t.length, first_t: motion.data.t[0], last_t: motion.data.t[motion.data.t.length - 1] }; })()")
                    print(f"record: err={rec_err} take={ {k: v for k, v in (take or {}).items() if k != 'meta'} }")
                    if rec_err or not take:
                        failures.append(f"record mode failed: {rec_err}")
                    else:
                        if take["n"] < 30 or take["n_channels"] != 28 or take["n_t"] != take["n"]:
                            failures.append(f"record: bad motion table ({take['n']} frames, {take['n_channels']} channels)")
                        if take["audioBytes"] <= 0:
                            failures.append("record: empty audio")
                        meta = take["meta"]
                        for k in ("source", "role", "neutral", "license", "tool_versions", "rate_hz", "audio"):
                            if k not in meta:
                                failures.append(f"record: meta.json missing '{k}'")
                        zb = base64.b64decode(ev("window.animacy.record.lastZipBase64()"))
                        with zipfile.ZipFile(io.BytesIO(zb)) as z:
                            names = z.namelist()
                            bad = z.testzip()
                            motion = json.loads(z.read(f"{take['name']}/motion.json"))
                            print(f"record zip: {len(zb)} bytes, {names}, testzip={bad}, motion schema={motion['schema']} n={motion['n']} rate={motion['rate_hz']}")
                            if bad or set(names) != {f"{take['name']}/motion.json", f"{take['name']}/audio.webm", f"{take['name']}/meta.json"}:
                                failures.append(f"record: zip contents wrong: {names} bad={bad}")
                            if motion["schema"] != "animacy.human.v1" or motion["n"] != len(motion["data"]["head_yaw"]):
                                failures.append("record: motion.json is not a canonical clip table")
                    shot("12_record_take")
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
