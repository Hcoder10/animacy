"""Voice activity -> per-frame ``speaking`` flag.

Two backends, chosen at runtime and recorded in ``meta.json['vad']``:

* ``silero-vad`` (pip package; the model ships inside the wheel, no torch.hub
  download needed). Used when importable.
* ``energy`` fallback: 40 ms RMS in dB with an adaptive noise floor (10th
  percentile), a two-threshold hysteresis gate (+12 dB on / +6 dB off) and a
  short dwell, so a single loud click does not become "speech" and a syllable
  gap does not drop the flag.

Limitation: any voice on the track counts. An off-camera interviewer's
question is flagged ``speaking=1`` on the on-camera subject's frames; single-
speaker sources (addresses, vlogs) avoid this, diarization would fix it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .capture_math import hysteresis, segments_to_mask


def _silero_segments(audio: np.ndarray, sr: int) -> Optional[List[Tuple[float, float]]]:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception:
        return None
    model = load_silero_vad()
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    ts = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True,
                               min_speech_duration_ms=150, min_silence_duration_ms=200)
    return [(float(d["start"]), float(d["end"])) for d in ts]


def energy_mask(audio: np.ndarray, sr: int, t_grid: np.ndarray, win_s: float = 0.04,
                on_db: float = 12.0, off_db: float = 6.0) -> np.ndarray:
    """Energy VAD sampled at the grid times. Pure numpy; see module docstring."""
    audio = np.asarray(audio, dtype=np.float32)
    half = int(win_s * sr / 2)
    idx = np.clip((np.asarray(t_grid) * sr).astype(int), 0, max(len(audio) - 1, 0))
    rms = np.array([np.sqrt(np.mean(audio[max(i - half, 0):i + half + 1] ** 2) + 1e-12) for i in idx])
    db = 20 * np.log10(rms + 1e-9)
    floor = float(np.percentile(db, 10))
    rate = 1.0 / max(float(t_grid[1] - t_grid[0]), 1e-6) if len(t_grid) > 1 else 30.0
    return hysteresis(db, floor + on_db, floor + off_db,
                      min_on=max(1, int(0.1 * rate)), min_off=max(1, int(0.25 * rate)))


def speaking_mask(audio: Optional[np.ndarray], sr: int, t_grid: np.ndarray) -> Tuple[np.ndarray, str]:
    """(bool mask over ``t_grid``, backend name)."""
    t_grid = np.asarray(t_grid, dtype=float)
    if audio is None or len(audio) < sr // 10:
        return np.zeros(len(t_grid), dtype=bool), "none"
    segs = _silero_segments(audio, sr)
    if segs is not None:
        try:
            import silero_vad

            ver = getattr(silero_vad, "__version__", "?")
        except Exception:
            ver = "?"
        return segments_to_mask(segs, t_grid), f"silero-vad {ver}"
    return energy_mask(audio, sr, t_grid), "energy"
