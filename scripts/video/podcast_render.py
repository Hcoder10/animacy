"""data/video/podcast/show.json -> the podcast takes, one MP4 per camera per section.

    # everything (camera A full show, then E open/close, then B/C/D per section)
    C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe scripts/video/podcast_render.py --all

    # just the wide, or just look at the set
    ... scripts/video/podcast_render.py --cam A
    ... scripts/video/podcast_render.py --stills
    ... scripts/video/podcast_render.py --probe

Same shape as ``animacy/grade/render.py``: a local ``http.server`` over the repo,
Chromium on ``web/podcast.html``, and every frame set explicitly — frame i of the
video is ``window.podcast.seek(i)``, which is row i of the show. Nothing is timed
by the browser, so a slow renderer only makes rendering slower, never the motion,
and the frame count of a take is exactly the frame count of its slice of the
show. That is what keeps the angles in sync when the editor cuts between them.

Its ``retry``/backoff, ``free_port``/``wait_port`` and ``ffmpeg_binary`` are
imported rather than re-written. The one thing done differently is the encode:
1920x1080 for four minutes is far too much to buffer as PNG files, so frames are
piped straight into ffmpeg as they are grabbed.

Playwright lives in C:/Users/sarta/reachy-duplex/.venv. Runs headed by default
(headed Chromium gets a real GPU context on this box; headless falls back to
SwiftShader, which is ~20x slower at this resolution).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from animacy.grade.render import ffmpeg_binary, free_port, retry, wait_port  # noqa: E402

WIDTH, HEIGHT, FPS = 1920, 1080, 30
CAMERAS = ("A", "B", "C", "D", "E")

_GRAB_JS = """
([indices, mime, quality]) => {
  const cv = document.querySelector('#stage canvas');
  const out = [];
  for (const i of indices) {
    window.podcast.seek(i);
    out.push(cv.toDataURL(mime, quality).split(',')[1]);
  }
  return out;
}
"""


# --------------------------------------------------------------------- browser
class PodcastPage:
    """Chromium on web/podcast.html, one frame at a time. Context manager."""

    def __init__(self, headless: bool = False, ready_timeout_ms: int = 240_000, log=print,
                 width: int = WIDTH, height: int = HEIGHT, show_url: Optional[str] = None):
        self.headless = headless
        self.ready_timeout_ms = ready_timeout_ms
        self.log = log
        self.width, self.height = width, height
        self.show_url = show_url
        self._srv = self._pw = self._browser = self._ctx = None
        self.page = None
        self.port = None
        self.info: Dict = {}
        self.gl = "?"
        self.console_errors: List[str] = []

    def __enter__(self) -> "PodcastPage":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self.port = free_port()
        self._srv = subprocess.Popen([sys.executable, "-m", "http.server", str(self.port), "--bind", "127.0.0.1"],
                                     cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_port(self.port)
        self._pw = sync_playwright().start()
        gpu = ["--ignore-gpu-blocklist", "--use-angle=d3d11", "--enable-gpu-rasterization",
               "--enable-zero-copy", "--hide-scrollbars"]
        cpu = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
               "--ignore-gpu-blocklist", "--hide-scrollbars"]
        if self.headless:
            self._browser = self._pw.chromium.launch(headless=True, args=cpu)
        else:
            # headed for the real GPU context; if there is no display to open a
            # window on, fall back to headless SwiftShader rather than dying
            try:
                self._browser = self._pw.chromium.launch(headless=False, args=gpu)
            except Exception as e:  # noqa: BLE001
                self.log(f"[page] no headed browser ({type(e).__name__}: {str(e)[:120]}); "
                         f"falling back to headless SwiftShader — expect ~20x slower")
                self.headless = True
                self._browser = self._pw.chromium.launch(headless=True, args=cpu)
        # the window is the frame: the canvas fills the viewport at pixel ratio 1
        self._ctx = self._browser.new_context(viewport={"width": self.width, "height": self.height},
                                              device_scale_factor=1)
        self.page = None

        def open_page():
            if self.page is not None:
                try:
                    self.page.close()
                except Exception:  # noqa: BLE001
                    pass
            self.page = self._ctx.new_page()
            self.page.on("console", lambda m: self.console_errors.append(m.text) if m.type == "error" else None)
            self.page.on("pageerror", lambda e: self.console_errors.append(f"pageerror: {e}"))
            q = f"?show={self.show_url}" if self.show_url else ""
            self.page.goto(f"http://127.0.0.1:{self.port}/web/podcast.html{q}", wait_until="domcontentloaded")
            # the URDFs and their meshes come off a CDN + disk; ready resolves when both are in
            self.page.wait_for_function("window.podcast && window.podcast.readyInfo",
                                        timeout=self.ready_timeout_ms)

        retry(open_page, attempts=4, backoff=(15.0, 45.0, 120.0), what="podcast page load", log=self.log)
        self.info = self.page.evaluate("window.podcast.info()")
        self.gl = self.page.evaluate(
            "() => { const c = document.createElement('canvas').getContext('webgl2');"
            "  const e = c && c.getExtension('WEBGL_debug_renderer_info');"
            "  return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; }")
        cw, ch = self.info["canvas"]["width"], self.info["canvas"]["height"]
        if (cw, ch) != (self.width, self.height):
            raise RuntimeError(f"canvas is {cw}x{ch}, expected {self.width}x{self.height}")
        if self.info["errors"]:
            raise RuntimeError(f"page reported errors: {self.info['errors']}")

    def close(self) -> None:
        for closer in (lambda: self._browser and self._browser.close(), lambda: self._pw and self._pw.stop(),
                       lambda: self._srv and self._srv.terminate()):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._browser = self._pw = self._srv = None

    def set_camera(self, cam: str, f0: int, f1: int) -> Dict:
        return self.page.evaluate("([c, a, b]) => window.podcast.setCamera(c, a, b)", [cam, f0, f1])

    def grab(self, indices: Sequence[int], mime: str = "image/png", quality: float = 0.98) -> List[bytes]:
        data = retry(lambda: self.page.evaluate(_GRAB_JS, [list(indices), mime, quality]),
                     attempts=3, backoff=(8.0, 25.0), what=f"grab {indices[0]}..{indices[-1]}", log=self.log)
        return [base64.b64decode(d) for d in data]

    def debug(self) -> Dict:
        return self.page.evaluate("window.podcast.debug()")


# --------------------------------------------------------------------- encode
def encode_take(page: PodcastPage, cam: str, f0: int, f1: int, out_mp4: str, narration: Optional[str],
                batch: int = 4, crf: int = 17, mime: str = "image/png", quality: float = 0.98,
                log=print) -> Dict:
    """Frames ``[f0, f1)`` on camera ``cam`` -> ``out_mp4``, piped into ffmpeg as they are grabbed.

    The audio is the matching slice of ``narration.wav``: the show clock is the
    frame clock, so the slice is exactly ``f0/FPS`` for ``(f1-f0)/FPS`` seconds."""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
    n = f1 - f0
    page.set_camera(cam, f0, f1)
    cmd = [ffmpeg_binary(), "-y", "-loglevel", "error",
           "-f", "image2pipe", "-framerate", str(FPS), "-i", "pipe:0"]
    if narration:
        cmd += ["-ss", f"{f0 / FPS:.6f}", "-t", f"{n / FPS:.6f}", "-i", narration, "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-movflags", "+faststart", out_mp4]
    t0 = time.perf_counter()
    # stderr to a file, not a pipe: nobody drains a pipe while we are busy
    # writing frames into stdin, and a full stderr buffer would deadlock the
    # encode partway through a four-minute take.
    errf = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
        written = 0
        try:
            for i in range(f0, f1, batch):
                idx = list(range(i, min(f1, i + batch)))
                for png in page.grab(idx, mime=mime, quality=quality):
                    proc.stdin.write(png)
                    written += 1
                if written % 300 == 0 or written == n:
                    el = time.perf_counter() - t0
                    rate = written / max(el, 1e-6)
                    log(f"    {written}/{n} frames  {rate:.1f} fps  "
                        f"eta {max(0.0, (n - written) / max(rate, 1e-6)):.0f}s", flush=True)
            proc.stdin.close()
        except Exception:
            proc.kill()
            raise
        rc = proc.wait()
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace").strip()
    finally:
        errf.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {err[:800]}")
    if written != n:
        raise RuntimeError(f"{out_mp4}: wrote {written} frames, expected {n}")
    el = time.perf_counter() - t0
    return {"camera": cam, "f_start": f0, "f_end": f1, "frames": n,
            "t_start": round(f0 / FPS, 3), "t_end": round(f1 / FPS, 3), "seconds": round(n / FPS, 3),
            "path": os.path.relpath(out_mp4, ROOT).replace("\\", "/"),
            "bytes": os.path.getsize(out_mp4), "render_seconds": round(el, 1),
            "render_fps": round(written / max(el, 1e-6), 2)}


# --------------------------------------------------------------------- plan
def take_plan(show: Dict, cams: Sequence[str]) -> List[Dict]:
    """What to render: the whole show wide, the open/close on the push-in, and
    every section on each single/over-the-shoulder angle."""
    n = show["n_frames"]
    secs = show["sections"]
    plan: List[Dict] = []
    if "A" in cams:
        plan.append({"camera": "A", "section": "full", "f0": 0, "f1": n})
    if "E" in cams and secs:
        plan.append({"camera": "E", "section": "open", "f0": 0, "f1": secs[0]["f_end"]})
        plan.append({"camera": "E", "section": "close", "f0": secs[-1]["f_start"], "f1": n})
    for cam in ("B", "C", "D"):
        if cam not in cams:
            continue
        for s in secs:
            plan.append({"camera": cam, "section": f"s{int(s['index']) + 1:02d}",
                         "f0": s["f_start"], "f1": s["f_end"], "title": s["title"],
                         "line_indices": s["line_indices"]})
    return plan


def line_indices_for(show: Dict, f0: int, f1: int) -> List[int]:
    return [ln["index"] for ln in show["lines"] if ln["f_start"] < f1 and ln["f_start"] + ln["f_count"] > f0]


# --------------------------------------------------------------------- stills
def pick_still_frames(show: Dict) -> List[Tuple[str, int]]:
    """A few frames worth looking at: mid-line on each host, and one section beat."""
    out: List[Tuple[str, int]] = []
    lines = show["lines"]

    def mid(ln):
        return ln["f_start"] + ln["f_count"] // 2

    lamp_lines = [ln for ln in lines if ln["host"] == "LAMP"]
    reachy_lines = [ln for ln in lines if ln["host"] == "REACHY"]
    if lamp_lines:
        out.append(("lamp_speaks", mid(lamp_lines[len(lamp_lines) // 3])))
        out.append(("lamp_speaks_2", mid(lamp_lines[-2 if len(lamp_lines) > 1 else 0])))
    if reachy_lines:
        out.append(("reachy_speaks", mid(reachy_lines[len(reachy_lines) // 3])))
        out.append(("reachy_speaks_2", mid(reachy_lines[-2 if len(reachy_lines) > 1 else 0])))
    out.append(("rest", 8))
    if len(show["sections"]) > 1:
        out.append(("section_beat", show["sections"][1]["f_start"] - 8))
    return out


def write_stills(page: PodcastPage, show: Dict, out_dir: str, cams: Sequence[str], log=print) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    picks = pick_still_frames(show)
    written = []
    for cam in cams:
        for name, f in picks:
            page.set_camera(cam, 0, show["n_frames"])
            png = page.grab([f])[0]
            p = os.path.join(out_dir, f"{cam}_{name}_f{f:05d}.png")
            with open(p, "wb") as fh:
                fh.write(png)
            written.append(p)
            log(f"  still {os.path.basename(p)}  {len(png) / 1024:.0f} KB")
    return written


# --------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "video", "podcast"))
    ap.add_argument("--cam", action="append", choices=list(CAMERAS), help="render only these cameras (repeatable)")
    ap.add_argument("--all", action="store_true", help="every camera (the default when --cam is not given)")
    ap.add_argument("--section", type=int, action="append", help="1-based section numbers only")
    ap.add_argument("--probe", action="store_true", help="print the set measurements and exit")
    ap.add_argument("--stills", action="store_true", help="write data/video/podcast/stills/*.png and exit")
    ap.add_argument("--bench", type=int, default=0, help="grab N frames and report the rate, then exit")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--jpeg", action="store_true", help="grab frames as JPEG (faster readback, slightly softer)")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="stop each take after N frames — a smoke test that OVERWRITES the real "
                         "take at that path, so re-render without it afterwards")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.dir)
    show_path = os.path.join(out_dir, "show.json")
    if not os.path.exists(show_path):
        print(f"no {show_path} — run scripts/video/show_build.py first")
        return 2
    with open(show_path, encoding="utf-8") as fh:
        show = json.load(fh)
    narration = None if args.no_audio else os.path.join(out_dir, show.get("narration_wav", "narration.wav"))
    if narration and not os.path.exists(narration):
        print(f"[warn] no narration at {narration}; rendering silent")
        narration = None
    cams = tuple(dict.fromkeys(args.cam)) if args.cam else CAMERAS
    mime, quality = ("image/jpeg", 0.95) if args.jpeg else ("image/png", 0.98)

    # show.json is served over the same http.server the page is on
    show_url = "../" + os.path.relpath(show_path, ROOT).replace("\\", "/")
    t_boot = time.perf_counter()
    with PodcastPage(headless=args.headless, show_url=show_url) as page:
        print(f"[page] {page.gl} | {'headless' if args.headless else 'headed'} | "
              f"{page.info['canvas']['width']}x{page.info['canvas']['height']} | "
              f"{page.info['n_frames']} frames ({page.info['seconds']:.1f}s) | boot {time.perf_counter() - t_boot:.1f}s")
        if show.get("placeholder_voice"):
            print("[page] NOTE: show.json was built on the placeholder voice takes")

        if args.probe:
            print(json.dumps(page.debug(), indent=2))
            return 0
        if args.bench:
            t0 = time.perf_counter()
            got = 0
            for i in range(0, args.bench, args.batch):
                got += len(page.grab(range(i, min(args.bench, i + args.batch)), mime=mime, quality=quality))
            el = time.perf_counter() - t0
            print(f"[bench] {got} frames in {el:.1f}s = {got / el:.2f} fps "
                  f"({show['n_frames'] / max(got / el, 1e-9) / 60:.1f} min per full-show pass)")
            return 0
        if args.stills:
            write_stills(page, show, os.path.join(out_dir, "stills"), cams)
            return 0

        plan = take_plan(show, cams)
        if args.section:
            # only the per-section takes; the whole-show wide and the open/close
            # are not sections, so asking for one section never re-renders them
            keep = {f"s{n:02d}" for n in args.section}
            plan = [p for p in plan if p["section"] in keep]
            if not plan:
                print(f"no takes for section(s) {args.section} on camera(s) {list(cams)}")
                return 2
        total = sum(p["f1"] - p["f0"] for p in plan)
        print(f"[plan] {len(plan)} takes, {total} frames ({total / FPS / 60:.1f} min of footage)")

        manifest_path = os.path.join(out_dir, "render_manifest.json")
        clips: List[Dict] = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as fh:
                    clips = json.load(fh).get("clips", [])
            except Exception:  # noqa: BLE001
                clips = []

        def save_manifest():
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"schema": "animacy.podcast.render.v1", "fps": FPS,
                           "size": [WIDTH, HEIGHT], "show": "show.json",
                           "narration": os.path.basename(narration) if narration else None,
                           "placeholder_voice": bool(show.get("placeholder_voice")),
                           "gl": page.gl,
                           "cameras": {
                               "A": "wide two-shot", "B": "medium single, LAMP", "C": "medium single, REACHY",
                               "D": "over the shoulder, behind LAMP onto REACHY",
                               "E": "wide with a 1 deg push-in (open/close)"},
                           "clips": sorted(clips, key=lambda c: (c["camera"], c["f_start"]))}, fh, indent=2)

        for k, p in enumerate(plan, 1):
            f0, f1 = p["f0"], p["f1"]
            if args.limit_frames:
                f1 = min(f1, f0 + args.limit_frames)
            out_mp4 = os.path.join(out_dir, p["camera"], f"{p['section']}.mp4")
            print(f"[{k}/{len(plan)}] cam {p['camera']} · {p['section']} · frames {f0}..{f1} "
                  f"({(f1 - f0) / FPS:.1f}s) -> {os.path.relpath(out_mp4, ROOT)}", flush=True)
            rec = encode_take(page, p["camera"], f0, f1, out_mp4, narration, batch=args.batch,
                              crf=args.crf, mime=mime, quality=quality)
            rec["section"] = p["section"]
            rec["title"] = p.get("title", "")
            rec["line_indices"] = line_indices_for(show, f0, f1)
            clips = [c for c in clips if not (c["camera"] == rec["camera"] and c["section"] == rec["section"])]
            clips.append(rec)
            save_manifest()
            print(f"      {rec['frames']} frames, {rec['bytes'] / 1e6:.1f} MB, "
                  f"{rec['render_fps']:.1f} fps render", flush=True)

        if page.console_errors:
            print(f"[warn] {len(page.console_errors)} console error(s): {page.console_errors[:3]}")
        print(f"[done] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
