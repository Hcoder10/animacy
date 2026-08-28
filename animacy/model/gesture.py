"""Gesture placement: one clear human gesture landing on each prosodic accent.

Continuous window matching produces sway by construction; a blind judge scores
the clips where "one clear gesture lands on the accent". This module

* finds the utterance's ACCENTS from the 30 Hz features (peaks of the smoothed
  log-energy envelope: 1 for < 2.5 s of speech, 2 for 2.5-5 s, 3 beyond, at least
  0.8 s apart; the first stressed peak after onset and the peak nearest the final
  word are always kept), plus the speech onset and end;
* takes a GESTURE from the index's library for the intent (top-K windows by
  prototype score, each with its own peak time = the frame of maximal head
  angular speed, and its duration), chosen by a seeded draw that prefers
  gestures fitting the gap to the next accent;
* PLACES it so its peak lands on the accent, at ``tier x 1.2`` amplitude, and
  BLENDS it over the base retrieval stream with raised-cosine crossfades
  (``blend_ms``); the base is damped to ``base_gain_active`` while a gesture is
  active and ``base_gain_idle`` elsewhere so the gesture reads. Thinking = one
  tilt-and-hold from the onset, held (base at ``hold_gain``) until the last accent,
  then released; excitement = the rise gesture on the first accent, held, dropped
  at the last accent. Everything stays inside the clip; rate limiting is downstream.

All numbers are mirrored in ``model.json: postprocess.gesture_placement`` so the
browser reproduces the placement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..schema import RATE_HZ
from .data import MODEL_CHANNELS

_HEAD = [MODEL_CHANNELS.index(c) for c in ("head_yaw", "head_pitch", "head_roll")]
QUIET_ENERGY = -0.3


@dataclass
class PlacementConfig:
    enabled: bool = True
    k: int = 25                    # library size per intent (top-K by prototype score)
    blend_ms: float = 150.0        # raised-cosine crossfade at each gesture edge
    base_gain_active: float = 0.6  # base stream gain while a gesture is active
    base_gain_idle: float = 0.8    # base stream gain elsewhere
    hold_gain: float = 0.4         # base gain during a thinking hold
    amplitude_boost: float = 1.2   # gesture amplitude = intent tier x this
    min_gap_s: float = 0.8         # minimum spacing between accents
    seed: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_any(cls, v) -> "PlacementConfig":
        if isinstance(v, PlacementConfig):
            return v
        if v is None or v is True:
            return cls()
        if v is False or v == 0:
            return cls(enabled=False)
        if isinstance(v, (int, float)):
            return cls(enabled=bool(v))
        d = dict(v)
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class Accents:
    onset: int
    end: int
    accents: List[int]
    speech_s: float
    envelope_peaks: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def detect_accents(features: np.ndarray, speaking: Optional[np.ndarray] = None, min_gap_s: float = 0.8,
                   rate_hz: float = RATE_HZ) -> Accents:
    """Prosodic accents of an utterance from feature 64 (normalised log energy)."""
    from scipy.signal import find_peaks

    f = np.asarray(features, np.float32)
    T = len(f)
    if T == 0:
        return Accents(0, 0, [], 0.0)
    energy = f[:, 64] if f.shape[1] > 64 else np.zeros(T, np.float32)
    voiced = (np.asarray(speaking) > 0) if speaking is not None and len(speaking) == T else (energy > QUIET_ENERGY)
    idx = np.nonzero(voiced)[0]
    if len(idx) == 0:
        return Accents(0, T, [], 0.0)
    onset, end = int(idx[0]), int(idx[-1]) + 1
    speech_s = (end - onset) / rate_hz
    n_acc = 1 if speech_s < 2.5 else (2 if speech_s < 5.0 else 3)
    k = np.hanning(9)
    k /= k.sum()
    env = np.convolve(energy, k, mode="same")
    env = np.where(voiced, env, env.min())
    dist = max(1, int(round(min_gap_s * rate_hz)))
    peaks, _ = find_peaks(env[onset:end], distance=dist, prominence=0.15)
    peaks = peaks + onset
    if len(peaks) == 0:                               # a flat utterance: its loudest frame
        peaks = np.array([onset + int(np.argmax(env[onset:end]))])
    heights = env[peaks]
    chosen: List[int] = []
    # the first stressed peak after the onset and the peak nearest the final word are kept
    first = int(peaks[0])
    last = int(peaks[-1])
    chosen.append(first)
    if n_acc >= 2 and last != first:
        chosen.append(last)
    rest = [int(p) for p in peaks[np.argsort(-heights)] if int(p) not in chosen]
    for p in rest:
        if len(chosen) >= n_acc:
            break
        if all(abs(p - c) >= dist for c in chosen):
            chosen.append(p)
    chosen = sorted(chosen)[:n_acc]
    return Accents(onset, end, chosen, speech_s, [int(p) for p in peaks])


def _raised_cosine(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), np.float32)
    return (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)


def _add(gest: np.ndarray, gain: np.ndarray, seg: np.ndarray, start: int, blend: int, gain_active: float,
         fade_in: bool = True, fade_out: bool = True) -> None:
    """Add ``seg`` into the gesture layer at ``start`` with raised-cosine edges (clipped to the
    clip bounds) and mark the base gain over that span."""
    T = len(gest)
    n = len(seg)
    s0, s1 = max(0, start), min(T, start + n)
    if s1 <= s0:
        return
    seg = seg[s0 - start: s1 - start]
    w = np.ones(s1 - s0, np.float32)
    b = min(blend, (s1 - s0) // 2)
    if b > 0:
        ramp = _raised_cosine(b)
        if fade_in and start >= 0:
            w[:b] = ramp
        if fade_out and start + n <= T:
            w[-b:] = ramp[::-1]
    gest[s0:s1] += seg * w[:, None]
    gain[s0:s1] = np.minimum(gain[s0:s1], gain_active)


def _hold(gest: np.ndarray, gain: np.ndarray, pose: np.ndarray, start: int, until: int, release: int, hold_gain: float) -> None:
    """Keep ``pose`` (a [14] excursion) from ``start`` to ``until``, then release it over ``release`` frames."""
    T = len(gest)
    s0, s1 = max(0, start), min(T, until)
    if s1 > s0:
        gest[s0:s1] += pose[None, :]
        gain[s0:s1] = np.minimum(gain[s0:s1], hold_gain)
    r1 = min(T, s1 + release)
    if r1 > s1:
        w = 1.0 - _raised_cosine(r1 - s1)
        gest[s1:r1] += pose[None, :] * w[:, None]


def place_gestures(base: np.ndarray, features: np.ndarray, speaking: Optional[np.ndarray], index, intent_tag: str,
                   tier: float, cfg: Optional[PlacementConfig] = None, rate_hz: float = RATE_HZ):
    """base [T, 14] (retrieval or model stream, canonical units) -> (motion [T, 14], placements).

    ``out = base * gain + gestures``: the base keeps swaying at ``base_gain_idle`` (0.8), is damped
    to ``base_gain_active`` (0.6) under a gesture and to ``hold_gain`` (0.4) during a hold, and each
    gesture (the library window's own excursion from its first frame, at tier x amplitude_boost)
    is added with raised-cosine edges so its peak lands exactly on the accent. ``index`` must
    carry a gesture library (``index.gestures[intent_tag]``: {id, peak_t, dur, score})."""
    cfg = PlacementConfig.from_any(cfg)
    m = np.array(base, np.float32, copy=True)
    T = len(m)
    info: Dict = {"enabled": cfg.enabled, "intent": intent_tag, "placements": []}
    lib = getattr(index, "gestures", None)
    if not cfg.enabled or T == 0 or not lib or intent_tag not in lib or not lib[intent_tag]:
        info["reason"] = "disabled" if not cfg.enabled else "no gesture library for this intent"
        return m, info
    acc = detect_accents(features, speaking, cfg.min_gap_s, rate_hz)
    info["accents"] = acc.to_dict()
    if not acc.accents:
        info["reason"] = "no accents"
        return m, info
    rng = np.random.default_rng(cfg.seed)
    blend = max(1, int(round(cfg.blend_ms / 1000.0 * rate_hz)))
    gain = np.full(T, cfg.base_gain_idle, np.float32)
    gest = np.zeros_like(m)
    amp = float(tier) * cfg.amplitude_boost
    entries = list(lib[intent_tag])

    def pick(gap_frames: Optional[int]):
        fitting = [e for e in entries if gap_frames is None or e["dur"] <= gap_frames]
        pool = fitting if fitting else sorted(entries, key=lambda e: e["dur"])[: max(1, len(entries) // 3)]
        return pool[int(rng.integers(len(pool)))]

    def segment(e):
        seg = index.gesture_segment(e["id"]).astype(np.float32)
        return (seg - seg[:1]) * amp                     # the gesture is its own excursion from where it started

    release = 2 * blend
    if intent_tag == "thinking":
        e = pick(None)
        seg = segment(e)
        peak = min(len(seg) - 1, int(e["peak_t"]))
        start = acc.onset
        _add(gest, gain, seg[: peak + 1], start, blend, cfg.base_gain_active, fade_out=False)
        hold_until = max(start + peak + 1, acc.accents[-1])
        _hold(gest, gain, seg[peak], start + peak + 1, hold_until, release, cfg.hold_gain)
        info["placements"].append({"kind": "tilt-and-hold", "gesture_id": int(e["id"]), "start": int(start), "peak_at": int(start + peak),
                                   "hold_until": int(hold_until), "release_frames": int(release), "score": float(e["score"])})
    elif intent_tag == "excitement":
        e = pick(None)
        seg = segment(e)
        peak = min(len(seg) - 1, int(e["peak_t"]))
        first, last = acc.accents[0], acc.accents[-1]
        start = first - peak
        _add(gest, gain, seg[: peak + 1], start, blend, cfg.base_gain_active, fade_out=False)
        hold_until = last if last > first else min(T, first + int(0.8 * rate_hz))
        _hold(gest, gain, seg[peak], first + 1, hold_until, release, cfg.hold_gain)
        info["placements"].append({"kind": "rise-hold-drop", "gesture_id": int(e["id"]), "start": int(start), "peak_at": int(first),
                                   "drop_at": int(hold_until), "release_frames": int(release), "score": float(e["score"])})
    else:
        for i, a in enumerate(acc.accents):
            gap = (acc.accents[i + 1] - a) if i + 1 < len(acc.accents) else None
            e = pick(gap)
            seg = segment(e)
            peak = min(len(seg) - 1, int(e["peak_t"]))
            start = a - peak
            _add(gest, gain, seg, start, blend, cfg.base_gain_active)
            info["placements"].append({"kind": "gesture", "gesture_id": int(e["id"]), "accent": int(a), "start": int(start),
                                       "peak_at": int(a), "dur": int(len(seg)), "score": float(e["score"])})
    out = m * gain[:, None] + gest
    info["base_gain"] = {"idle": cfg.base_gain_idle, "active": cfg.base_gain_active, "hold": cfg.hold_gain}
    info["gesture_amplitude"] = amp
    return out.astype(np.float32), info


def library_from_index(index, k: int = 25, tags: Sequence[str] = ("agreement", "doubt", "excitement", "thinking", "greeting")) -> Dict[str, List[Dict]]:
    """Top-K windows per intent by prototype score, each with its peak frame (max head angular
    speed over the window + continuation) and duration in frames."""
    out: Dict[str, List[Dict]] = {}
    if index.proto is None:
        return out
    for t in tags:
        if t not in index.proto:
            continue
        order = np.argsort(-index.proto[t])[:k]
        entries = []
        for i in order:
            seg = index.gesture_segment(int(i))
            v = np.linalg.norm(np.diff(seg[:, _HEAD], axis=0), axis=1)
            peak = int(np.argmax(v)) + 1 if len(v) else 0
            entries.append({"id": int(i), "peak_t": peak, "dur": int(len(seg)), "score": round(float(index.proto[t][i]), 3)})
        out[t] = entries
    return out
