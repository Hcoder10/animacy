"""Robot joint table -> MP4 through the three.js viewer (Playwright + ffmpeg).

The web viewer (``web/``) is the only renderer that draws both robots' real
URDFs, so the grader reuses it headlessly: a local ``http.server`` serves the
repo, Chromium opens ``web/``, and for every frame of a joint table this module
sets the URDF joint values on ``window.animacy.robots[<robot>].viewer``
directly (no animation loop, no wall clock), renders, and pulls the canvas as
PNG. ffmpeg then encodes 30 fps H.264 with the utterance audio muxed in.

Nothing here is timed by the browser: frame ``i`` of the video is row ``i`` of
the joint table resampled to ``fps``, so a slow software renderer only makes
rendering slower, never the motion.

Every clip starts with a title card ("Clip N" + the transcript line) whose
text the caller chooses; the card is the only text in the video.
"""
from __future__ import annotations

import base64
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from html import escape
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..profile import Profile
from ..retarget import to_urdf_values

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FPS = 30
CARD_BG = "#0e1117"     # the viewer's scene background, so the card cuts to the clip without a flash
DEFAULT_SIZE = 512
VIEW = "iso"            # 3/4 front (the viewer's 'iso': +38 deg azimuth from the robot's front, slightly above)


def retry(fn, attempts: int = 3, backoff: Sequence[float] = (15.0, 60.0, 120.0), what: str = "", log=None,
          exceptions=(Exception,), sleep=time.sleep):
    """Call ``fn()`` up to ``attempts`` times, sleeping ``backoff[i]`` after failure ``i``. Transient network or
    browser stalls (a CDN import that hangs, a screenshot that waits on fonts during an outage, a judge call that
    times out) must not kill a run that has hours of cached work behind it."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:  # noqa: PERF203
            last = e
            if i == attempts - 1:
                break
            wait = backoff[min(i, len(backoff) - 1)]
            if log:
                log(f"[retry] {what or getattr(fn, '__name__', 'call')}: attempt {i + 1}/{attempts} failed "
                    f"({type(e).__name__}: {str(e)[:160]}); retrying in {wait:.0f}s")
            sleep(wait)
    raise last


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 15.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"http server did not come up on {port}")


def ffmpeg_binary() -> str:
    b = shutil.which("ffmpeg")
    if not b:
        raise RuntimeError("ffmpeg not on PATH")
    return b


def resample_table(table: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Any ``t``-indexed joint table -> a uniform ``fps`` grid (linear interpolation, hold at the ends)."""
    t = table["t"].to_numpy(dtype=np.float64)
    t = t - t[0]
    dur = float(t[-1]) if len(t) else 0.0
    n = max(1, int(np.floor(dur * fps + 1e-6)) + 1)
    tn = np.arange(n) / fps
    out = {"t": tn}
    for c in table.columns:
        if c == "t":
            continue
        out[c] = np.interp(tn, t, table[c].to_numpy(dtype=np.float64))
    return pd.DataFrame(out)


def complete_table(table: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Fill joints the table does not carry with the profile's rest value."""
    out = table.copy()
    for j in profile.joints:
        if j.name not in out.columns:
            out[j.name] = j.rest
    return out


def urdf_frames(table: pd.DataFrame, profile: Profile) -> List[Dict[str, float]]:
    """Per-frame ``{urdf_joint: value}`` dicts (radians / metres)."""
    vals = to_urdf_values(complete_table(table, profile), profile)
    n = len(table)
    return [{k: float(v[i]) for k, v in vals.items()} for i in range(n)]


_HIDE_CSS = """
.topbar, .controls, .readouts, #webcam-thumb, .vp-label, .vp-sub, .vp-loading, #toast, footer { display: none !important; }
body { display: block !important; background: %(bg)s !important; }
.viewports { display: flex !important; gap: 0 !important; background: %(bg)s !important; }
.viewport { width: %(size)dpx !important; height: %(size)dpx !important; flex: 0 0 %(size)dpx !important; min-height: %(size)dpx !important; }
.viewport.ab { display: none !important; }
"""

_CAPTURE_JS = """
([name, frames]) => {
  const app = window.animacy;
  const R = app.robots[name];
  if (!R) throw new Error('robot not loaded: ' + name);
  const cv = R.viewer.renderer.domElement;
  const out = [];
  for (const vals of frames) {
    R.viewer.setJoints(vals);
    R.viewer.render();
    out.push(cv.toDataURL('image/png').split(',')[1]);
  }
  return out;
}
"""

_SETUP_JS = """
([zoom, view]) => {
  const app = window.animacy;
  // the animation loop must not touch the joints: detach every source
  if (app.source && app.source.stop) { try { app.source.stop(); } catch (e) {} }
  app.source = null;
  app.channels = null;
  const info = {};
  for (const name of Object.keys(app.robots)) {
    const R = app.robots[name];
    R.viewer.controls.enableDamping = false;
    R.viewer.resize();
    R.viewer.frame((R.profile.description.viewer && R.profile.description.viewer.camera_distance) || null);
    R.viewer.bounds.distance *= zoom;
    R.viewer.setView(view);
    R.viewer.render();
    const cv = R.viewer.renderer.domElement;
    info[name] = { width: cv.width, height: cv.height, standin: !!R.standin, missing: R.missingJoints,
                   urdfJoints: R.viewer.jointNames };
  }
  return info;
}
"""


class ViewerRenderer:
    """Headless renderer around the web viewer. Use as a context manager.

    >>> with ViewerRenderer() as r:
    ...     r.render_clip("lamp", table, profile, "out.mp4", title="Clip 3", subtitle="The robot says: hi")
    """

    def __init__(self, size: int = DEFAULT_SIZE, zoom: float = 1.15, headless: bool = True, gpu: bool = False,
                 view: str = VIEW, ready_timeout_ms: int = 180_000, log=print):
        self.log = log
        self.size = size
        self.zoom = zoom
        self.headless = headless
        self.gpu = gpu
        self.view = view
        self.ready_timeout_ms = ready_timeout_ms
        self._srv = None
        self._pw = None
        self._browser = None
        self._ctx = None
        self.page = None
        self.card_page = None
        self.port = None
        self.info: Dict = {}
        self.console_errors: List[str] = []

    # ---- lifecycle ------------------------------------------------------------
    def __enter__(self) -> "ViewerRenderer":
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
        if self.gpu:
            args = ["--ignore-gpu-blocklist", "--use-angle=d3d11", "--enable-gpu-rasterization",
                    "--autoplay-policy=no-user-gesture-required"]
        else:
            args = ["--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--autoplay-policy=no-user-gesture-required"]
        self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
        self._ctx = self._browser.new_context(viewport={"width": 2 * self.size + 64, "height": self.size + 64},
                                              device_scale_factor=1)
        self.page = None

        def open_viewer():
            # a fresh page per attempt: the viewer's CDN imports (three, urdf-loader) fail hard during a network blip
            if self.page is not None:
                try:
                    self.page.close()
                except Exception:  # noqa: BLE001
                    pass
            self.page = self._ctx.new_page()
            self.page.on("console", lambda m: self.console_errors.append(m.text) if m.type == "error" else None)
            self.page.on("pageerror", lambda e: self.console_errors.append(f"pageerror: {e}"))
            self.page.goto(f"http://127.0.0.1:{self.port}/web/?autoplay=0&source=native", wait_until="domcontentloaded")
            self.page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=self.ready_timeout_ms)

        retry(open_viewer, attempts=4, backoff=(20.0, 60.0, 180.0), what="viewer page load", log=self.log)
        self.page.add_style_tag(content=_HIDE_CSS % {"bg": CARD_BG, "size": self.size})
        self.page.wait_for_timeout(400)  # ResizeObserver -> renderer.setSize
        self.info = self.page.evaluate(_SETUP_JS, [self.zoom, self.view])
        for name, i in self.info.items():
            if (i["width"], i["height"]) != (self.size, self.size):
                raise RuntimeError(f"{name}: canvas is {i['width']}x{i['height']}, expected {self.size}x{self.size}")
            if i["standin"]:
                raise RuntimeError(f"{name}: the viewer loaded a stand-in URDF; the real one must be present for grading")
        self.card_page = self._ctx.new_page()
        self.card_page.set_viewport_size({"width": self.size, "height": self.size})

    def close(self) -> None:
        for closer in (lambda: self._browser and self._browser.close(), lambda: self._pw and self._pw.stop(),
                       lambda: self._srv and self._srv.terminate()):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._browser = self._pw = self._srv = None

    # ---- frames -----------------------------------------------------------------
    def robots(self) -> List[str]:
        return list(self.info)

    def render_frames(self, robot: str, table: pd.DataFrame, profile: Profile, fps: float = FPS,
                      batch: int = 12) -> List[bytes]:
        """PNG bytes per frame of ``table`` (resampled to ``fps``) on ``robot``."""
        if robot not in self.info:
            raise KeyError(f"robot {robot!r} not loaded in the viewer (have {self.robots()})")
        frames = urdf_frames(resample_table(table, fps), profile)
        out: List[bytes] = []
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            data = retry(lambda: self.page.evaluate(_CAPTURE_JS, [robot, chunk]), attempts=3, backoff=(10.0, 30.0),
                         what=f"frame capture {robot} {i}", log=self.log)
            out.extend(base64.b64decode(d) for d in data)
        return out

    def card_png(self, title: str, subtitle: str = "", footnote: str = "") -> bytes:
        html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
        html, body {{ margin: 0; width: {self.size}px; height: {self.size}px; background: {CARD_BG};
                      color: #e6e9ef; font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        .wrap {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
                 justify-content: center; text-align: center; padding: 0 40px; box-sizing: border-box; }}
        .t {{ font-size: {int(self.size * 0.13)}px; font-weight: 700; letter-spacing: 0.02em; margin-bottom: 26px; }}
        .s {{ font-size: {int(self.size * 0.045)}px; line-height: 1.35; color: #cfd5e1; max-width: {int(self.size * 0.86)}px; }}
        .f {{ position: absolute; bottom: 18px; left: 0; right: 0; font-size: {int(self.size * 0.03)}px; color: #6f7890; }}
        </style></head><body><div class="wrap"><div class="t">{escape(title)}</div>
        <div class="s">{escape(subtitle)}</div></div><div class="f">{escape(footnote)}</div></body></html>"""
        def shoot():
            self.card_page.set_content(html, wait_until="load")
            return self.card_page.screenshot(type="png", timeout=60_000, animations="disabled", caret="hide")

        return retry(shoot, attempts=4, backoff=(15.0, 60.0, 180.0), what="card render", log=self.log)

    def black_png(self) -> bytes:
        return self.card_png("", "")

    # ---- encode ---------------------------------------------------------------------
    def render_clip(self, robot: str, table: pd.DataFrame, profile: Profile, out_mp4: str, title: str,
                    subtitle: str = "", audio: Optional[np.ndarray] = None, sr: int = 16000,
                    card_seconds: float = 1.0, fps: float = FPS, work_dir: Optional[str] = None) -> Dict:
        """Card + frames + (optional) audio -> ``out_mp4``. Returns timing/size facts."""
        t0 = time.perf_counter()
        frames = self.render_frames(robot, table, profile, fps)
        t_render = time.perf_counter() - t0
        card = self.card_png(title, subtitle)
        n_card = int(round(card_seconds * fps))
        encode_frames([card] * n_card + frames, out_mp4, fps=fps, audio=audio, sr=sr, audio_offset=card_seconds,
                      work_dir=work_dir)
        return {"frames": len(frames), "card_frames": n_card, "seconds": (n_card + len(frames)) / fps,
                "render_seconds": t_render, "fps_render": len(frames) / max(t_render, 1e-6),
                "bytes": os.path.getsize(out_mp4)}


def _write_audio(path: str, audio: Optional[np.ndarray], sr: int, total_seconds: float, offset: float) -> None:
    import soundfile as sf

    n_total = int(round(total_seconds * sr))
    buf = np.zeros(n_total, dtype=np.float32)
    if audio is not None and len(audio):
        a = np.asarray(audio, dtype=np.float32)
        start = int(round(offset * sr))
        end = min(n_total, start + len(a))
        if end > start:
            buf[start:end] = a[: end - start]
    sf.write(path, buf, sr)


def encode_frames(frames: Sequence[bytes], out_mp4: str, fps: float = FPS, audio: Optional[np.ndarray] = None,
                  sr: int = 16000, audio_offset: float = 0.0, work_dir: Optional[str] = None, crf: int = 20) -> str:
    """PNG frames (+ audio, always present as a stream so reels concatenate) -> H.264/AAC MP4."""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="animacy_frames_", dir=work_dir)
    try:
        for i, png in enumerate(frames):
            with open(os.path.join(tmp, f"{i:06d}.png"), "wb") as fh:
                fh.write(png)
        wav = os.path.join(tmp, "audio.wav")
        _write_audio(wav, audio, sr, len(frames) / fps, audio_offset)
        cmd = [ffmpeg_binary(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%06d.png"),
               "-i", wav, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-r", str(fps), "-c:a", "aac", "-b:a", "64k", "-ar", str(sr), "-ac", "1", "-shortest", out_mp4]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def concat_mp4(parts: Sequence[str], out_mp4: str, fps: float = FPS, crf: int = 20) -> str:
    """Concatenate clips (re-encoded, so mixed encoder settings never break the reel)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)) or ".", exist_ok=True)
    fd, lst = tempfile.mkstemp(suffix=".txt", prefix="concat_")
    os.close(fd)
    try:
        with open(lst, "w", encoding="utf-8") as fh:
            for p in parts:
                fh.write("file '" + os.path.abspath(p).replace("\\", "/").replace("'", r"'\''") + "'\n")
        cmd = [ffmpeg_binary(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
               "-c:a", "aac", "-b:a", "64k", out_mp4]
        subprocess.run(cmd, check=True)
    finally:
        os.remove(lst)
    return out_mp4




def contact_sheet(frames: Sequence[bytes], out_png: str, n: int = 12, cols: int = 6) -> str:
    """Fallback for a judge that cannot watch video: ``n`` evenly spaced frames on one sheet."""
    from io import BytesIO

    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, num=min(n, len(frames))).round().astype(int)
    tiles = [Image.open(BytesIO(frames[i])).convert("RGB") for i in idx]
    w, h = tiles[0].size
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), CARD_BG)
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    sheet.save(out_png)
    return out_png


def joint_plot(table: pd.DataFrame, profile: Profile, out_png: str, title: str = "") -> str:
    """Fallback companion: every joint against time (robot units), no text beyond joint names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
    for j in profile.joints:
        if j.name in table.columns:
            ax.plot(table["t"], table[j.name] - j.rest, label=j.name, lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint - rest")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
