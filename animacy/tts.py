"""Local text-to-speech that returns the WAVEFORM (needed for the motion model).

Windows: System.Speech (SAPI) via PowerShell, rendered to a WAV file — no
downloads, no keys. Elsewhere: ``espeak-ng`` if present. Output is resampled to
16 kHz mono float32 so it feeds ``animacy.features.audio_features`` directly.
Kokoro (the browser demo's voice) can be used here too via ``kokoro-onnx`` when
installed; it is optional because it needs a 300 MB model download.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple

import numpy as np


def _resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def synth_sapi(text: str, rate: int = 0, voice: str | None = None) -> Tuple[np.ndarray, int]:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    safe = text.replace("'", "''")
    sel = f"$s.SelectVoice('{voice}'); " if voice else ""
    cmd = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           f"{sel}$s.Rate = {int(rate)}; $s.SetOutputToWaveFile('{path}'); $s.Speak('{safe}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    os.remove(path)
    return _resample_to_16k(data.mean(axis=1), sr), 16000


def synth_espeak(text: str) -> Tuple[np.ndarray, int]:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run(["espeak-ng", "-w", path, text], check=True)
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    os.remove(path)
    return _resample_to_16k(data.mean(axis=1), sr), 16000


def synth_kokoro(text: str, voice: str = "af_heart") -> Tuple[np.ndarray, int]:
    from kokoro_onnx import Kokoro  # optional dependency

    model = os.environ.get("KOKORO_ONNX", "kokoro-v1.0.onnx")
    voices = os.environ.get("KOKORO_VOICES", "voices-v1.0.bin")
    k = Kokoro(model, voices)
    samples, sr = k.create(text, voice=voice, speed=1.0, lang="en-us")
    return _resample_to_16k(np.asarray(samples, dtype=np.float32), sr), 16000


def synth(text: str, engine: str = "auto") -> Tuple[np.ndarray, int]:
    """Return (wav16k float32 mono, 16000)."""
    if engine == "kokoro":
        return synth_kokoro(text)
    if engine == "sapi" or (engine == "auto" and sys.platform == "win32"):
        return synth_sapi(text)
    if shutil.which("espeak-ng"):
        return synth_espeak(text)
    raise RuntimeError("no TTS engine available (Windows SAPI, espeak-ng, or kokoro-onnx)")


def play_async(wav16k: np.ndarray):
    """Start playback on the default output device and return immediately."""
    import sounddevice as sd

    sd.play(wav16k, 16000, blocking=False)
    return sd
