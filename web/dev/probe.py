"""Evaluate a JS expression against the running viewer (dev helper).

    python web/dev/probe.py "window.animacy.robotInfo('lamp')"
    python web/dev/probe.py --query standin=1 "window.animacy.robots.lamp.viewer.bounds"
    python web/dev/probe.py --shot out.png "1"        # also save a screenshot

Starts its own http.server + headless Chromium like screenshot.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screenshot import ROOT, free_port, wait_port  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expr", nargs="+")
    ap.add_argument("--query", default="autoplay=0")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--wait", type=int, default=500, help="ms to wait after ready")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                              "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera", "microphone"]).new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            page.goto(f"http://127.0.0.1:{port}/web/?{a.query}", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            page.wait_for_timeout(a.wait)
            for expr in a.expr:
                try:
                    val = page.evaluate(f"(async () => ({expr}))()")
                    print(f">>> {expr}\n{json.dumps(val, indent=1, default=str)}")
                except Exception as e:  # noqa: BLE001
                    print(f">>> {expr}\nERROR {e}")
            if a.shot:
                page.wait_for_timeout(300)
                page.screenshot(path=a.shot)
                print("shot", a.shot)
            for l in logs:
                print(l[:400])
            browser.close()
    finally:
        srv.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
