"""Shared plumbing for the B-roll shots (scripts/video/broll_*.py).

Every clip is 1920x1080 / 30 fps / h264 / silent, written to data/video/broll/
with an entry in data/video/broll/manifest.json saying what it shows and which
command or URL produced it.

Nothing here fabricates content: the terminal shots replay stdout that was
actually captured from the real command (see `run_capture`), the viewer shots
drive the real page, and the browser shots visit the real URL.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.path.join(ROOT, "data", "video", "broll")
WORK_DIR = os.path.join(ROOT, "data", "video", "_work")
MANIFEST = os.path.join(OUT_DIR, "manifest.json")
BROLL_WEB = os.path.join(ROOT, "web", "dev", "broll")

FPS = 30
W, H = 1920, 1080
PY = sys.executable
VENV_BIN = os.path.dirname(PY)
# the installed console script, so the terminal shows the command a user types
ANIMACY = os.path.join(VENV_BIN, "animacy.exe")
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    print(msg, flush=True)


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not on PATH")
    return exe


def ffprobe_bin() -> str:
    exe = shutil.which("ffprobe") or ffmpeg_bin().replace("ffmpeg", "ffprobe")
    return exe


# ---------------------------------------------------------------------------
# local http server (the viewer + the b-roll pages are served from the repo root)
# ---------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(port: int, timeout: float = 20.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"http server did not come up on {port}")


class Server:
    """`python -m http.server` rooted at the repo, for the lifetime of a `with`."""

    def __init__(self, root: str = ROOT):
        self.root = root
        self.port = free_port()
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [PY, "-m", "http.server", str(self.port), "--bind", "127.0.0.1"],
            cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_port(self.port)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}/{path.lstrip('/')}"


# ---------------------------------------------------------------------------
# frame capture
# ---------------------------------------------------------------------------
class Capture:
    """Screenshot-per-frame capture of a page that is advanced manually.

    Frame i is exactly i/FPS of clip time no matter how slow the renderer is,
    because the page is stepped by a fixed dt rather than by wall clock.
    """

    def __init__(self, page, frames_dir: str, quality: int = 92):
        self.page = page
        self.dir = frames_dir
        self.n = 0
        self.quality = quality
        self.t_shot = 0.0
        os.makedirs(frames_dir, exist_ok=True)

    def grab(self, clip: dict | None = None) -> None:
        t0 = time.perf_counter()
        self.page.screenshot(path=os.path.join(self.dir, f"{self.n:05d}.jpg"),
                             type="jpeg", quality=self.quality, clip=clip, animations="allow")
        self.t_shot += time.perf_counter() - t0
        self.n += 1

    def step(self, frames: int, js: str | None = None, dt: float = 1.0 / FPS, clip: dict | None = None) -> None:
        for _ in range(frames):
            if js:
                self.page.evaluate(js.replace("$DT", repr(dt)))
            self.grab(clip=clip)

    def step_until(self, js_step: str, js_done: str, max_frames: int,
                   dt: float = 1.0 / FPS, clip: dict | None = None) -> int:
        k = 0
        while k < max_frames:
            self.page.evaluate(js_step.replace("$DT", repr(dt)))
            self.grab(clip=clip)
            k += 1
            if self.page.evaluate(js_done):
                break
        return k

    def hold(self, seconds: float, clip: dict | None = None) -> None:
        """Repeat the last frame (a still beat) without stepping the page."""
        if self.n == 0:
            raise RuntimeError("hold() before any frame")
        src = os.path.join(self.dir, f"{self.n - 1:05d}.jpg")
        for _ in range(int(round(seconds * FPS))):
            shutil.copyfile(src, os.path.join(self.dir, f"{self.n:05d}.jpg"))
            self.n += 1

    @property
    def seconds(self) -> float:
        return self.n / FPS


# ---------------------------------------------------------------------------
# encode + manifest
# ---------------------------------------------------------------------------
def encode(frames_dir: str, out_path: str, *, crf: int = 20, fps: int = FPS,
           vf: str | None = None, preset: str = "medium") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scale = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1117"
    chain = scale if not vf else f"{vf},{scale}"
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, "%05d.jpg"), "-vf", chain,
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-r", str(fps), "-an", "-movflags", "+faststart", os.path.abspath(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def probe(path: str) -> dict:
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "format=duration:stream=width,height,codec_name,avg_frame_rate",
                          "-of", "json", path], capture_output=True, text=True)
    aud = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    has_audio = bool(aud.stdout.strip())
    try:
        j = json.loads(out.stdout)
        st = (j.get("streams") or [{}])[0]
        return {"duration_s": round(float(j["format"]["duration"]), 2),
                "width": st.get("width"), "height": st.get("height"),
                "codec": st.get("codec_name"), "fps": st.get("avg_frame_rate"),
                "has_audio": has_audio, "bytes": os.path.getsize(path)}
    except Exception:  # noqa: BLE001
        return {"has_audio": has_audio, "bytes": os.path.getsize(path)}


def black_windows(path: str, min_seconds: float = 0.25, ymax_th: int = 60) -> list[dict]:
    """Stretches with nothing on screen, as [{start, end, seconds}] in clip time.

    A cut that lands inside one of these goes black mid-sentence. The grading
    reels legitimately contain them — they are the spacers between judged clips —
    so the answer is to publish where they are, not to remove them.

    ffmpeg's `blackdetect` is useless here: these clips are pillarboxed onto a
    near-black background, so the padding alone satisfies "most pixels are dark"
    and a perfectly legible title card counts as black. Peak luma per frame
    (signalstats YMAX) asks the question that actually matters — is there a
    bright pixel anywhere? — and separates a real spacer from a dark card.
    """
    out = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", path, "-an",
         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YMAX:file=-",
         "-f", "null", "-"], capture_output=True, text=True)
    times: list[tuple[float, float]] = []
    t = None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            for part in line.split():
                if part.startswith("pts_time:"):
                    t = float(part.split(":", 1)[1])
        elif "signalstats.YMAX=" in line and t is not None:
            times.append((t, float(line.split("=", 1)[1])))
    if not times:
        return []
    step = (times[-1][0] - times[0][0]) / max(1, len(times) - 1)
    wins, start = [], None
    for i, (ts, ymax) in enumerate(times):
        dark = ymax < ymax_th
        if dark and start is None:
            start = ts
        elif not dark and start is not None:
            if ts - start >= min_seconds:
                wins.append({"start": round(start, 2), "end": round(ts, 2),
                             "seconds": round(ts - start, 2)})
            start = None
    if start is not None:
        end = times[-1][0] + step
        if end - start >= min_seconds:
            wins.append({"start": round(start, 2), "end": round(end, 2),
                         "seconds": round(end - start, 2)})
    return wins


def register(filename: str, *, section: str, shows: str, source: str,
             notes: str | None = None, extra: dict | None = None,
             supplied: bool = False) -> dict:
    """Add/replace this clip's manifest entry (keyed by filename), keeping order.

    `supplied=True` marks footage handed to the edit rather than produced here:
    it is kept untouched and is not held to the 6-15 s / silent format rules.
    """
    path = os.path.join(OUT_DIR, filename)
    entry = {"file": filename, "section": section, "shows": shows, "source": source,
             "supplied": supplied}
    entry.update(probe(path))
    # where a cut would land on black; the edit needs this to pick an in-point
    entry["black_windows"] = black_windows(path)
    if entry["black_windows"]:
        log(f"  note: {len(entry['black_windows'])} black window(s): "
            + ", ".join(f"{w['start']}-{w['end']}s" for w in entry["black_windows"]))
    if notes:
        entry["notes"] = notes
    if extra:
        entry.update(extra)
    entry["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    data = {"format": {"width": W, "height": H, "fps": FPS, "codec": "h264", "audio": "none"},
            "note": "Every clip is real: real stdout, the real viewer, the real dataset pages, the "
                    "real robot's read-back. Clips with supplied=false were produced by "
                    "scripts/video/broll_*.py at 1920x1080 / 30 fps / h264 / silent, 6-15 s. Clips "
                    "with supplied=true are footage handed to the edit, kept untouched. "
                    "See docs/video/broll.md.",
            "clips": []}
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001
            pass
    clips = [c for c in data.get("clips", []) if c.get("file") != filename]
    clips.append(entry)
    clips.sort(key=lambda c: (c.get("section", ""), c.get("file", "")))
    data["clips"] = clips
    data["updated_utc"] = entry["recorded_utc"]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    log(f"  registered {filename}: {entry.get('duration_s')} s, "
        f"{entry.get('width')}x{entry.get('height')}, {entry.get('bytes', 0) / 1e6:.1f} MB")
    return entry


def workdir(name: str) -> str:
    d = os.path.join(WORK_DIR, name)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# real command capture (terminal shots)
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """One prompt + command + its ACTUAL stdout/stderr."""
    command: str
    output: str
    cwd: str = "~/animacy"
    exit_code: int = 0
    pause_after: float = 0.9
    seconds: float | None = None       # measured wall time of the real run
    output_delay: float = 0.35         # beat between the Enter key and the first output line
    output_cps: float = 900.0          # how fast the captured output paints


def run_capture(command: str, *, cwd: str = ROOT, display: str | None = None,
                pause_after: float = 0.9, timeout: int = 900, env: dict | None = None,
                max_lines: int | None = None) -> Block:
    """Run `command` for real and keep exactly what it printed.

    `display` is what the terminal page shows on the prompt line (e.g. the
    `python -m animacy.cli ...` form of a call we invoke via the venv python).
    """
    log(f"  $ {display or command}")
    t0 = time.time()
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", COLUMNS="118", NO_COLOR="1")
    if env:
        e.update(env)
    p = subprocess.run(command, cwd=cwd, shell=True, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout, env=e)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    out = out.replace("\r\n", "\n").rstrip("\n")
    if max_lines is not None:
        lines = out.split("\n")
        if len(lines) > max_lines:
            out = "\n".join(lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"])
    log(f"    exit {p.returncode} in {dt:.1f} s, {len(out.splitlines())} lines")
    return Block(command=display or command, output=out, exit_code=p.returncode,
                 pause_after=pause_after, seconds=round(dt, 2))


def run_bash(command: str, *, cwd: str = ROOT, pause_after: float = 0.9,
             timeout: int = 900, max_lines: int | None = None) -> Block:
    """Run a POSIX one-liner in Git Bash from the repo root; display it verbatim."""
    if not os.path.exists(GIT_BASH):
        raise SystemExit(f"Git Bash not found at {GIT_BASH}")
    posix_root = "/" + cwd.replace(":", "").replace("\\", "/")
    wrapped = f'"{GIT_BASH}" -c "cd {posix_root} && {command}"'
    b = run_capture(wrapped, cwd=cwd, display=command, pause_after=pause_after,
                    timeout=timeout, max_lines=max_lines)
    return b


@dataclass
class TermShot:
    """A terminal clip: a sequence of real blocks typed out at a human pace."""
    blocks: list[Block] = field(default_factory=list)
    title: str = "animacy"
    type_cps: float = 21.0             # keystrokes per second on the command line
    tail_hold: float = 1.4             # still beat on the final frame

    def payload(self) -> dict:
        return {"title": self.title, "typeCps": self.type_cps,
                "blocks": [{"command": b.command, "output": b.output, "cwd": b.cwd,
                            "exitCode": b.exit_code, "pauseAfter": b.pause_after,
                            "outputDelay": b.output_delay, "outputCps": b.output_cps}
                           for b in self.blocks]}


def record_term(shot: TermShot, out_name: str, *, section: str, shows: str, source: str,
                notes: str | None = None, headless: bool = True,
                max_seconds: float = 15.0) -> dict:
    """Render a TermShot through web/dev/broll/term.html and encode it."""
    from playwright.sync_api import sync_playwright

    work = workdir(out_name.replace(".mp4", ""))
    frames = os.path.join(work, "frames")
    out_path = os.path.join(OUT_DIR, out_name)
    with Server() as srv, sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--force-color-profile=srgb",
                                                             "--font-render-hinting=none"])
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.set_default_timeout(120_000)
        page.goto(srv.url("web/dev/broll/term.html"), wait_until="networkidle")
        page.wait_for_function("window.term && window.term.ready === true")
        page.evaluate("(d) => window.term.init(d)", shot.payload())
        cap = Capture(page, frames)
        cap.step_until("window.term.stepFrame($DT)", "window.term.done",
                       max_frames=int(max_seconds * FPS))
        cap.hold(shot.tail_hold)
        browser.close()
    encode(frames, out_path)
    entry = register(out_name, section=section, shows=shows, source=source, notes=notes)
    shutil.rmtree(work, ignore_errors=True)
    return entry


def record_doc(text: str, out_name: str, *, path_label: str, meta: str = "",
               section: str, shows: str, source: str, notes: str | None = None,
               px_per_sec: float = 46.0, plain: bool = False, no_gutter: bool = False,
               hold: float = 1.0, headless: bool = True, max_seconds: float = 15.0,
               fit: bool = False, seconds: float | None = None) -> dict:
    """Slow-scroll a real file's text through web/dev/broll/doc.html.

    `fit=True` shrinks the type until the whole excerpt is on screen and holds it
    still, for tables that want reading rather than travelling.
    """
    from playwright.sync_api import sync_playwright

    work = workdir(out_name.replace(".mp4", ""))
    frames = os.path.join(work, "frames")
    out_path = os.path.join(OUT_DIR, out_name)
    payload = {"text": text, "path": path_label, "meta": meta, "pxPerSec": px_per_sec,
               "plain": plain, "noGutter": no_gutter, "hold": hold, "fit": fit,
               "seconds": seconds}
    with Server() as srv, sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--force-color-profile=srgb",
                                                             "--font-render-hinting=none"])
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.set_default_timeout(120_000)
        page.goto(srv.url("web/dev/broll/doc.html"), wait_until="networkidle")
        page.wait_for_function("window.doc && window.doc.ready === true")
        page.evaluate("(d) => window.doc.init(d)", payload)
        info = page.evaluate("({total: window.doc.total, scrollS: window.doc.scrollS, "
                             "maxOffset: window.doc.maxOffset, fontPx: window.doc.fontPx || null})")
        log(f"  doc: needs {info['total']:.1f} s (scroll {info['maxOffset']:.0f} px)"
            + (f", type {info['fontPx']:.1f} px" if info.get("fontPx") else ""))
        if info["total"] > max_seconds:
            log(f"  WARNING: the shot would need {info['total']:.1f} s but is capped at "
                f"{max_seconds:.1f} s — it will stop mid-scroll")
        cap = Capture(page, frames)
        cap.step_until("window.doc.stepFrame($DT)", "window.doc.done",
                       max_frames=int(max_seconds * FPS))
        browser.close()
    encode(frames, out_path)
    entry = register(out_name, section=section, shows=shows, source=source, notes=notes)
    shutil.rmtree(work, ignore_errors=True)
    return entry
