"""Register externally-supplied footage of the physical robot, and cut an excerpt.

    python scripts/video/broll_hardware.py

`data/video/broll/real_reachy_desk_*.mp4` were filmed on a phone by a person,
not produced by this pipeline. This script never modifies them: it adds them to
the manifest so the edit knows they exist and where they came from, and writes a
separate `s6_reachy_hardware.mp4` — the busiest window of the 16:9 take, cut to
the film's format (1920x1080, 30 fps, h264, silent) so section 6 has a hero shot
inside the 6-15 s spec.

The window is chosen by frame-difference energy: real motion, picked
mechanically rather than by eye.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from broll_common import FPS, H, OUT_DIR, W, ffmpeg_bin, log, probe, register  # noqa: E402

SUPPLIED = {
    "real_reachy_desk_16x9.mp4":
        "Phone footage of the physical Reachy Mini Wireless on a desk, 16:9 — the actual "
        "hardware moving, filmed by a person. Supplied for the edit, not produced by this "
        "pipeline.",
    "real_reachy_desk_portrait.mp4":
        "The same physical Reachy Mini, portrait take (HEVC, has an audio track). Supplied "
        "for the edit, not produced by this pipeline.",
}


def motion_profile(path: str, step: int = 5, width: int = 240) -> tuple[np.ndarray, float]:
    """Per-sample frame-difference energy, and the seconds each sample covers."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, energy = None, []
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                h = int(frame.shape[0] * width / frame.shape[1])
                small = cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                energy.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
                prev = small
        i += 1
    cap.release()
    return np.asarray(energy), step / fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real_reachy_desk_16x9.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="s6_reachy_hardware.mp4")
    a = ap.parse_args()

    for name, shows in SUPPLIED.items():
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            log(f"  {name} is not here; skipping its manifest entry")
            continue
        info = probe(path)
        register(name, section="6", shows=shows, supplied=True,
                 source="filmed on a phone by a person; placed in data/video/broll/ for the edit",
                 notes=f"Untouched original ({info.get('duration_s')} s, {info.get('codec')}). "
                       "Not produced by scripts/video/broll_*.py, and its provenance is only what "
                       "the file itself shows: verify before using it as evidence of a specific run.")

    src = os.path.join(OUT_DIR, a.source)
    if not os.path.exists(src):
        log(f"  no {a.source} to excerpt")
        return 0
    energy, dt = motion_profile(src)
    width = max(2, int(round(a.seconds / dt)))
    if len(energy) <= width:
        start = 0.0
    else:
        sums = np.convolve(energy, np.ones(width), mode="valid")
        start = float(np.argmax(sums) * dt)
    log(f"  {a.source}: busiest {a.seconds:g} s starts at {start:.1f} s "
        f"(frame-difference energy over {len(energy)} samples)")

    out_path = os.path.join(OUT_DIR, a.out)
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                    "-i", src, "-t", f"{a.seconds:.2f}",
                    "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0e1117,fps={FPS}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", out_path], check=True)
    register(a.out, section="6",
             shows="The physical Reachy Mini Wireless moving on a desk — the real hardware, not a "
                   "render. Cut from the supplied 16:9 phone take at its busiest stretch, "
                   "conformed to 1920x1080 / 30 fps and silenced.",
             source=f"data/video/broll/{a.source}, {start:.1f}-{start + a.seconds:.1f} s "
                    f"(window chosen by frame-difference energy)",
             notes="Derived from footage this pipeline did not produce; the original is kept "
                   "untouched alongside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
