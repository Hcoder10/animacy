"""Audio for capture: extract from video files, or record the mic on a shared clock.

* :func:`extract_audio` — ``ffmpeg -i in -vn -ac 1 -ar 16000 -f wav`` via the
  ``ffmpeg`` on PATH, else imageio-ffmpeg's bundled binary if that package is
  installed. Returns ``(audio, backend)``; ``backend`` is recorded in meta.
* :class:`MicRecorder` — sounddevice input stream at 16 kHz on its own thread.
  ``t0`` is the ``time.perf_counter()`` at which sample 0 was captured, so
  video frames stamped with ``perf_counter()`` share the clock.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from .schema import AUDIO_SR


def _ffmpeg_binary() -> Tuple[Optional[str], str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe, "ffmpeg (PATH)"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg bundled"
    except Exception:
        return None, "none"


def extract_audio(video_path: str, sr: int = AUDIO_SR, max_seconds: float = 0.0) -> Tuple[Optional[np.ndarray], str]:
    """Mono float32 at ``sr`` from a video file, or (None, reason)."""
    import soundfile as sf

    exe, backend = _ffmpeg_binary()
    if exe is None:
        return None, "no ffmpeg available"
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [exe, "-v", "error", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr)]
    if max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-f", "wav", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 100:
            return None, f"{backend}: {r.stderr.strip()[-200:] or 'no audio stream'}"
        data, got_sr = sf.read(tmp, dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), backend
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class MicRecorder:
    """Background 16 kHz mono mic capture with a perf_counter start stamp."""

    def __init__(self, sr: int = AUDIO_SR, device=None) -> None:
        self.sr = sr
        self.device = device
        self.t0: Optional[float] = None
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        def cb(indata, frames, time_info, status):
            now = time.perf_counter()
            with self._lock:
                if self.t0 is None:
                    # first callback: sample 0 was captured `frames` samples ago
                    self.t0 = now - frames / self.sr
                self._chunks.append(indata[:, 0].copy())

        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=self.device,
                                      blocksize=512, callback=cb)
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            return np.concatenate(self._chunks).astype(np.float32) if self._chunks else np.zeros(0, np.float32)

    def slice(self, audio: np.ndarray, t_start: float, duration: float) -> np.ndarray:
        """Cut ``audio`` (returned by :meth:`stop`) to the video's clock window."""
        if self.t0 is None:
            return audio
        i0 = int(round((t_start - self.t0) * self.sr))
        n = int(round(duration * self.sr))
        out = np.zeros(n, np.float32)
        a0, a1 = max(i0, 0), min(i0 + n, len(audio))
        if a1 > a0:
            out[a0 - i0:a1 - i0] = audio[a0:a1]
        return out
