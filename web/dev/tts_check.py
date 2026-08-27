"""Headed check of the real Talk pipeline: Kokoro TTS in the page → model → robots.

    python web/dev/tts_check.py [--text "..."] [--backend model|retrieval|envelope] [--headless]

Downloads Kokoro-82M (~90 MB, cached by the browser profile afterwards), says
the line, reports timings / backend / frame count / audio-clock progress and
writes web/dev/shots/tts_talk.png. Exit 1 if say() fails or the robots do not move.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screenshot import ROOT, SHOTS, free_port, wait_port  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hi! I'm animacy: one motion space for any expressive robot. Watch us move while I talk.")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--profile", default=os.path.join(ROOT, "web", "dev", ".chromium-profile"), help="persistent profile so the TTS download is cached between runs")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    os.makedirs(SHOTS, exist_ok=True)
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = 0
    try:
        wait_port(port)
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(a.profile, headless=a.headless, viewport={"width": 1440, "height": 900},
                                                       args=["--autoplay-policy=no-user-gesture-required", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist"])
            page = ctx.new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}") if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            page.goto(f"http://127.0.0.1:{port}/web/?autoplay=0", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            page.evaluate("window.animacy.setSource('talk')")
            page.wait_for_function("window.animacy.sourceInfo().kind === 'talk'", timeout=20_000)
            if a.backend:
                page.evaluate(f"document.getElementById('talk-backend').value = '{a.backend}'")
            page.evaluate(f"document.getElementById('talk-voice').value = '{a.voice}'")
            print(f"backends available: {page.evaluate('window.animacy.backends.available()')}")
            t0 = time.time()
            last_status = ""
            page.evaluate(f"window.__said = null; window.animacy.say({a.text!r}).then(r => {{ window.__said = r || {{failed: true}}; }})")
            while page.evaluate("window.__said") is None:
                s = page.evaluate("document.getElementById('status').textContent")
                if s != last_status:
                    print(f"  [{time.time() - t0:6.1f}s] {s}")
                    last_status = s
                if time.time() - t0 > 900:
                    print("timeout waiting for say()")
                    rc = 1
                    break
                page.wait_for_timeout(500)
            said = page.evaluate("window.__said")
            print(f"say() → {said} after {time.time() - t0:.1f} s")
            if not said or said.get("failed"):
                rc = 1
            page.wait_for_timeout(1500)
            info = page.evaluate("window.animacy.talkInfo()")
            lamp = page.evaluate("window.animacy.getJointValues('lamp')")
            reachy = page.evaluate("window.animacy.getJointValues('reachy_mini')")
            print(f"audio clock t={info['time']:.2f}/{info['duration']:.2f} playing={info['playing']} backend={info['backend']}")
            print(f"lamp={ {k: round(v, 1) for k, v in lamp.items()} }")
            print(f"reachy={ {k: round(v, 1) for k, v in reachy.items()} }")
            moved = abs(lamp["wrist_pitch"] + 62.4) > 0.5 or abs(reachy["head_pitch"]) > 0.3 or abs(reachy["antenna_left"]) > 0.5
            if not moved:
                print("robots did not move")
                rc = 1
            page.screenshot(path=os.path.join(SHOTS, "tts_talk.png"))
            page.wait_for_timeout(max(0, int((info["duration"] - info["time"]) * 1000)) + 300)
            for l in logs[:20]:
                print(l[:300])
            ctx.close()
    finally:
        srv.terminate()
    print("OK" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
