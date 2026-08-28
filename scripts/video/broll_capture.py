"""B-roll for section 2: the real `animacy capture --preview` window, screen-recorded.

    python scripts/video/broll_capture.py [--seconds 12] [--source <video>]

Runs the actual tracker over a licensed clip from data/raw and captures the
OpenCV preview window with ffmpeg's gdigrab, so what is on screen is the
tracker's own overlay on a real person, at whatever rate the tracker really
manages. Nothing is re-drawn or re-timed.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import (  # noqa: E402
    ANIMACY, FPS, OUT_DIR, ROOT, ffmpeg_bin, ffprobe_bin, log, register,
)

WINDOW = "animacy capture (q to stop)"
# Public domain (White House); the same address that ships as web/clips/obama_2015_02_07.json
DEFAULT_SOURCE = os.path.join(ROOT, "data", "raw", "2015_02_07_President_Obama_s_Weekly_Address.webm")

user32 = ctypes.windll.user32


def find_window(title: str) -> int:
    return int(user32.FindWindowW(None, title) or 0)


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def wait_for_window(title: str, timeout: float = 180.0) -> int:
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = find_window(title)
        if h:
            _, _, w, hgt = window_rect(h)
            if w > 200 and hgt > 200:
                return h
        time.sleep(0.4)
    raise RuntimeError(f"the preview window {title!r} never appeared "
                       "(no interactive desktop, or capture failed to start)")


def video_size(path: str) -> tuple[int, int]:
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                         capture_output=True, text=True, check=True)
    w, h = out.stdout.strip().split("x")[:2]
    return int(w), int(h)


def drawn_crop(recording: str, source: str) -> str:
    """Where the preview image actually sits inside the captured window.

    OpenCV's window is DPI-unaware, so on a scaled display Windows reports a
    client area larger than the bitmap OpenCV blits into it: the frame lands at
    the top-left at its own pixel size (``draw_overlay`` copies the frame and
    never resizes it), and the rest of the client area is whatever was behind.
    """
    sw, sh = video_size(source)
    rw, rh = video_size(recording)
    w, h = min(sw, rw), min(sh, rh)
    if (w, h) != (sw, sh):
        log(f"  note: the {sw}x{sh} preview image is wider than the {rw}x{rh} grab; cropping to {w}x{h}")
    return f"crop={w - w % 2}:{h - h % 2}:0:0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--seconds", type=float, default=12.0, help="length of the recorded clip")
    ap.add_argument("--duration", type=float, default=90.0, help="seconds of video the tracker runs over")
    ap.add_argument("--settle", type=float, default=6.0,
                    help="seconds to let the tracker run before recording starts")
    ap.add_argument("--out", default="s2_capture_preview.mp4")
    ap.add_argument("--clip-out", default=os.path.join(ROOT, "out", "broll_capture_preview"))
    a = ap.parse_args()

    if not os.path.exists(a.source):
        raise SystemExit(f"source video not found: {a.source}")
    if find_window(WINDOW):
        raise SystemExit(f"a window called {WINDOW!r} is already open — close it first")
    shutil.rmtree(a.clip_out, ignore_errors=True)

    cmd = [ANIMACY, "capture", "--source", a.source, "-o", a.clip_out,
           "--preview", "--duration", str(a.duration), "--no-audio"]
    log("  launching: animacy capture --source data/raw/%s --preview --duration %g"
        % (os.path.basename(a.source), a.duration))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    raw = os.path.join(ROOT, "data", "video", "_work", "capture_preview.mp4")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    try:
        hwnd = wait_for_window(WINDOW)
        x, y, w, h = window_rect(hwnd)
        log(f"  preview window {w}x{h} at ({x},{y}); letting the tracker settle {a.settle:g} s")
        time.sleep(a.settle)                  # past the model warm-up and the clip's opening frames
        log(f"  recording {a.seconds:g} s with gdigrab")
        rec = subprocess.run(
            [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "gdigrab", "-framerate", str(FPS),
             "-draw_mouse", "0", "-i", f"title={WINDOW}", "-t", str(a.seconds),
             # gdigrab hands back the window's physical pixel size, which h264 needs even
             "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", raw],
            capture_output=True, text=True)
        if rec.returncode != 0:
            raise RuntimeError(f"gdigrab failed: {rec.stderr.strip()[:400]}")
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=20)
            tail = "\n".join((out or "").strip().splitlines()[-3:])
            if tail:
                log(f"  capture said: {tail}")
        except subprocess.TimeoutExpired:
            proc.kill()

    # OpenCV's window is DPI-unaware, so the drawn image sits in the top-left of the
    # captured window rect with black around it. Find that region and keep only it.
    crop = drawn_crop(raw, a.source)
    log(f"  cropping the recording to the drawn image: {crop}")
    out_path = os.path.join(OUT_DIR, a.out)
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", raw,
                    "-vf", f"{crop},scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
                           "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0e1117",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-r", str(FPS), "-an", "-movflags", "+faststart", out_path], check=True)
    os.remove(raw)
    register(a.out, section="2",
             shows="The real `animacy capture --preview` window tracking a licensed talking-head "
                   "clip: the tracker's landmark dots on an actual person's eyes, brows and nose, "
                   "with its live readout above them — clock, head yaw/pitch/roll and translation, "
                   "gaze, brow, eye-open, mouth, smile and torso lean, updating every frame.",
             source=f"animacy capture --source data/raw/{os.path.basename(a.source)} --preview "
                    f"(screen-recorded with ffmpeg gdigrab)",
             notes="Source: 2015-02-07 President Obama's Weekly Address, Public Domain "
                   "(commons.wikimedia.org). Window captured at its native size and scaled to 1080p.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
