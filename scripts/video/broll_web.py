"""B-roll of real web pages: the Hugging Face datasets, the live site, the repo.

    python scripts/video/broll_web.py --shots hf hf_large site github

Recorded with Playwright's own video recorder at 1920x1080, in real time, with
the pointer moved in steps and the page scrolled a little at a time — so the
motion is a person's pace rather than a jump cut. These are the live pages; the
only thing this script controls is where the cursor goes.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import (  # noqa: E402
    FPS, H, OUT_DIR, ROOT, W, ffmpeg_bin, log, register, workdir,
)

HF = "https://huggingface.co/datasets/squaredcuber/animacy-human-motion"
HF_LARGE = "https://huggingface.co/datasets/squaredcuber/animacy-human-motion-large"
SITE = "https://hcoder10.github.io/animacy/web/"
REPO = "https://github.com/Hcoder10/animacy"


class Recorder:
    """A Chromium context that records video of everything it does."""

    def __init__(self, p, name: str, *, gpu: bool = False):
        self.work = workdir(f"web_{name}")
        args = ["--autoplay-policy=no-user-gesture-required", "--ignore-gpu-blocklist",
                "--hide-scrollbars=false"]
        args += (["--use-angle=d3d11", "--enable-gpu-rasterization"] if gpu
                 else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        self.browser = p.chromium.launch(headless=True, args=args)
        self.ctx = self.browser.new_context(
            viewport={"width": W, "height": H}, device_scale_factor=1,
            record_video_dir=self.work, record_video_size={"width": W, "height": H})
        self.page = self.ctx.new_page()
        self.page.set_default_timeout(120_000)
        self.t0 = None

    def open(self, url: str, wait: str = "load") -> None:
        log(f"  opening {url}")
        self.page.goto(url, wait_until=wait)
        self.t0 = time.time()

    def glide(self, x: float, y: float, steps: int = 26, pause: float = 0.25) -> None:
        """Move the pointer there over `steps`, the way a hand does."""
        self.page.mouse.move(x, y, steps=steps)
        self.page.wait_for_timeout(int(pause * 1000))

    def creep(self, total_px: int, *, per: int = 44, ms: int = 55) -> None:
        """Scroll `total_px` in small wheel steps: readable, never a jump."""
        done = 0
        while done < total_px:
            step = min(per, total_px - done)
            self.page.mouse.wheel(0, step)
            self.page.wait_for_timeout(ms)
            done += step

    def finish(self, out_name: str, *, start: float, seconds: float, **meta) -> dict:
        path = self.page.video.path()
        self.ctx.close()                      # flushes the webm
        self.browser.close()
        src = path if os.path.exists(path) else (glob.glob(os.path.join(self.work, "*.webm")) + [None])[0]
        if not src:
            raise RuntimeError("Playwright wrote no video")
        out_path = os.path.join(OUT_DIR, out_name)
        subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                        "-i", src, "-t", f"{seconds:.2f}",
                        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1117,fps={FPS}",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-an", "-movflags", "+faststart", out_path], check=True)
        entry = register(out_name, **meta)
        shutil.rmtree(self.work, ignore_errors=True)
        return entry


def shot_dataset(p, url: str, out_name: str, label: str, section: str, shows: str) -> dict:
    r = Recorder(p, out_name.replace(".mp4", ""))
    r.open(url, wait="domcontentloaded")
    r.page.wait_for_timeout(3500)             # the card, the viewer and the file list settle
    title = r.page.title()
    log(f"  page title: {title}")
    r.glide(980, 420, steps=30, pause=0.6)
    r.creep(520, per=40, ms=60)               # down through the dataset card
    r.page.wait_for_timeout(900)
    r.creep(560, per=40, ms=60)
    r.page.wait_for_timeout(900)
    r.glide(1400, 620, steps=30, pause=0.8)
    r.creep(420, per=36, ms=60)
    r.page.wait_for_timeout(1400)
    return r.finish(out_name, start=3.2, seconds=13.0, section=section, shows=shows,
                    source=url, extra={"page_title": title})


def shot_site(p) -> dict:
    """The published viewer: load it, then actually drive it."""
    r = Recorder(p, "site", gpu=False)
    r.open(SITE, wait="domcontentloaded")
    try:
        r.page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=180_000)
        log("  the published viewer reached ready")
    except Exception:  # noqa: BLE001
        log("  WARNING: window.animacy.ready never became true on the live site")
    r.page.wait_for_timeout(1800)
    # switch to the canonical (human) clips and play one, at a human pace
    tab = r.page.locator("#source-tabs button[data-source='canonical']")
    b = tab.bounding_box()
    r.glide(b["x"] - 300, b["y"] - 220, steps=20, pause=0.5)
    r.glide(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, steps=28, pause=0.7)
    tab.click()
    r.page.wait_for_timeout(2600)
    sel = r.page.locator("#clip-select")
    sb = sel.bounding_box()
    r.glide(sb["x"] + 220, sb["y"] + sb["height"] / 2, steps=26, pause=0.6)
    r.page.wait_for_timeout(4200)             # let the robots move on the clip
    r.glide(1500, 300, steps=34, pause=1.6)
    clip = r.page.evaluate("window.animacy && window.animacy.sourceInfo()")
    log(f"  live site playing: {clip}")
    return r.finish("s9_live_site.mp4", start=2.0, seconds=14.0, section="9",
                    shows="The published site loading and being used: both robots on their real "
                          "URDFs, the source switched to a human canonical clip, and the retarget "
                          "running in the browser.",
                    source=SITE, extra={"playing": clip})


def shot_github(p) -> dict:
    r = Recorder(p, "github")
    r.open(REPO, wait="domcontentloaded")
    r.page.wait_for_timeout(3200)
    title = r.page.title()
    log(f"  page title: {title}")
    r.glide(760, 380, steps=30, pause=0.7)
    r.creep(460, per=38, ms=60)               # the file tree
    r.page.wait_for_timeout(900)
    r.creep(620, per=38, ms=60)               # into the README
    r.page.wait_for_timeout(1600)
    r.glide(1200, 560, steps=28, pause=1.2)
    return r.finish("s9_github_repo.mp4", start=2.6, seconds=12.0, section="9",
                    shows="The public repository: the robots/, animacy/ and docs/ tree and the top "
                          "of the README, on github.com.",
                    source=REPO, extra={"page_title": title})


SHOTS = {
    "hf": lambda p: shot_dataset(
        p, HF, "s7_hf_dataset.mp4", "animacy-human-motion", "7",
        "The Hugging Face dataset page for squaredcuber/animacy-human-motion: the card, the "
        "licence, and the files, scrolled at reading pace."),
    "hf_large": lambda p: shot_dataset(
        p, HF_LARGE, "s7_hf_dataset_large.mp4", "animacy-human-motion-large", "7",
        "The larger Hugging Face dataset, squaredcuber/animacy-human-motion-large: the same card "
        "structure at the scale the harvester is filling."),
    "site": shot_site,
    "github": shot_github,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="*", default=list(SHOTS), choices=list(SHOTS))
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from playwright.sync_api import sync_playwright

    failed = []
    with sync_playwright() as p:
        for name in a.shots:
            log(f"\n=== {name} ===")
            try:
                SHOTS[name](p)
            except Exception as e:  # noqa: BLE001
                log(f"  FAILED {name}: {type(e).__name__}: {e}")
                failed.append(name)
    if failed:
        log(f"\nfailed shots: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
