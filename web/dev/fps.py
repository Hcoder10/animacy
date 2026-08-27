"""Measure the viewer's render FPS in a real (headed, GPU) Chromium.

    python web/dev/fps.py            # opens a window for ~20 s, prints fps per scenario

Headless SwiftShader numbers from screenshot.py are not representative; this
uses the machine's GPU like a laptop would.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screenshot import ROOT, free_port, wait_port  # noqa: E402


def main() -> int:
    from playwright.sync_api import sync_playwright

    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--use-fake-device-for-media-capture", "--use-fake-ui-for-media-stream"])
            page = browser.new_context(viewport={"width": 1440, "height": 900}).new_page()
            page.goto(f"http://127.0.0.1:{port}/web/", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            scenarios = [
                ("native lamp headshake", "(async () => { await animacy.setSource('native'); await animacy.setClip('lamp/headshake'); })()"),
                ("canonical cal_talk, both robots", "(async () => { await animacy.setSource('canonical'); await animacy.setClip('synth/cal_talk'); })()"),
                ("canonical + A/B (3 viewports)", "(async () => { await animacy.setAb(true); })()"),
            ]
            for name, js in scenarios:
                page.evaluate(js)
                page.wait_for_timeout(1500)
                samples = []
                for _ in range(4):
                    page.wait_for_timeout(1000)
                    samples.append(page.evaluate("animacy.fps"))
                print(f"{name}: {sum(samples) / len(samples):.1f} fps (samples {[round(s) for s in samples]})")
            gpu = page.evaluate("(() => { const c = document.createElement('canvas'); const gl = c.getContext('webgl2') || c.getContext('webgl'); const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()")
            print("renderer:", gpu)
            browser.close()
    finally:
        srv.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
