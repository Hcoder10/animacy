"""Inference: audio features -> codes -> motion -> a ``HumanClip``.

Sampling per step: ``logits / temperature + bigram_weight * log P(code | prev)``
then a categorical draw from a seeded generator, so the same inputs and seed
give the same motion. The VQ decoder turns codes into 30 Hz standardised
frames, which are de-standardised, zero-phase smoothed, and dropped into a
neutral canonical frame table (``face_valid=1``, ``speaking`` copied).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .a2m_ar import AudioToMotionAR

from ..schema import BOUNDS, HumanClip, RATE_HZ, empty_frames
from .a2m import AudioToMotion
from .data import FRAMES_PER_CODE, MODEL_CHANNELS, pool_flag, pool_pairs
from .vq import MotionVQVAE

DEFAULT_SMOOTH_HZ = 6.0


@dataclass
class MotionModel:
    vq: MotionVQVAE
    a2m: Optional[AudioToMotion]                 # feed-forward audio -> codes ("ff")
    bigram_logp: Optional[np.ndarray]            # [n_codes, n_codes] log P(next | prev), for "ff" sampling
    info: Dict
    ar: Optional["AudioToMotionAR"] = None       # autoregressive audio -> codes ("ar")

    @classmethod
    def load(cls, ckpt_dir: str, device: str = "cpu") -> "MotionModel":
        vq = MotionVQVAE.load(os.path.join(ckpt_dir, "vq.pt"), device)
        a2m, bigram, ar = None, None, None
        if os.path.exists(os.path.join(ckpt_dir, "a2m.pt")):
            a2m, ck = AudioToMotion.load(os.path.join(ckpt_dir, "a2m.pt"), device)
            b = ck["bigram_logp"]
            bigram = np.asarray(b.numpy() if hasattr(b, "numpy") else b, np.float32)
        if os.path.exists(os.path.join(ckpt_dir, "a2m_ar.pt")):
            from .a2m_ar import AudioToMotionAR

            ar, _ = AudioToMotionAR.load(os.path.join(ckpt_dir, "a2m_ar.pt"), device)
        if a2m is None and ar is None:
            raise FileNotFoundError(f"no a2m.pt or a2m_ar.pt in {ckpt_dir}")
        info_path = os.path.join(ckpt_dir, "model_info.json")
        info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else {}
        return cls(vq=vq, a2m=a2m, bigram_logp=bigram, info=info, ar=ar)

    @property
    def n_codes(self) -> int:
        return self.vq.n_codes

    @property
    def archs(self):
        return [a for a, m in (("ff", self.a2m), ("ar", self.ar)) if m is not None]


def sample_codes(logits: np.ndarray, bigram_logp: Optional[np.ndarray], temperature: float = 0.8,
                 bigram_weight: float = 0.5, seed: int = 0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """[L, n_codes] logits -> [L] codes. Deterministic given ``seed``."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    lg = np.asarray(logits, np.float64) / max(float(temperature), 1e-6)
    L, n = lg.shape
    codes = np.zeros(L, np.int64)
    prev = -1
    for t in range(L):
        z = lg[t]
        if prev >= 0 and bigram_logp is not None and bigram_weight > 0:
            z = z + bigram_weight * bigram_logp[prev]
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        prev = int(rng.choice(n, p=p))
        codes[t] = prev
    return codes


def smooth_motion(motion: np.ndarray, rate_hz: float = RATE_HZ, cutoff_hz: Optional[float] = DEFAULT_SMOOTH_HZ) -> np.ndarray:
    """Zero-phase Butterworth per channel (no lag); short inputs are returned untouched."""
    x = np.asarray(motion, np.float64)
    if not cutoff_hz or len(x) < 12:
        return x.astype(np.float32)
    from scipy.signal import butter, filtfilt

    b, a = butter(2, min(cutoff_hz / (0.5 * rate_hz), 0.99))
    return filtfilt(b, a, x, axis=0, padlen=min(9, len(x) - 1)).astype(np.float32)


def motion_to_clip(motion_raw: np.ndarray, speaking: Optional[np.ndarray] = None, rate_hz: float = RATE_HZ,
                   **meta) -> HumanClip:
    """[T, 14] canonical-unit motion -> a valid HumanClip (neutral elsewhere)."""
    m = np.asarray(motion_raw, np.float32)
    T = len(m)
    frames = empty_frames(T, rate_hz)
    for i, c in enumerate(MODEL_CHANNELS):
        lo, hi = BOUNDS[c]
        frames[c] = np.clip(m[:, i], lo, hi)
    frames["face_valid"] = 1.0
    frames["arm_valid"] = 0.0
    if speaking is not None and len(speaking):
        s = np.asarray(speaking, np.float32)
        if len(s) != T:
            s = np.pad(s, (0, max(0, T - len(s))), mode="edge")[:T]
        frames["speaking"] = (s > 0).astype(np.float32)
    meta.setdefault("source", "model")
    return HumanClip.from_frames(frames, rate_hz=rate_hz, **meta)


def generate_motion(model: MotionModel, features: np.ndarray, speaking: np.ndarray, causal: bool = False,
                    temperature: float = 0.8, bigram_weight: float = 0.5, seed: int = 0,
                    smooth_hz: Optional[float] = DEFAULT_SMOOTH_HZ, arch: str = "ff",
                    top_p: float = 0.9, repeat_penalty: float = 0.0, stay_bias: float = 0.0,
                    stay_energy: float = -0.3) -> Tuple[np.ndarray, np.ndarray]:
    """[T, 66], [T] on the 30 Hz grid -> (motion [T, 14] raw units, codes [T//2]).
    ``arch`` = "ff" (per-step logits + bigram prior) or "ar" (autoregressive: temperature, top-p,
    repeat penalty, stay bias when the audio is quieter than ``stay_energy``)."""
    f = np.asarray(features, np.float32)
    s = np.asarray(speaking, np.int64)
    T = len(f)
    if T < FRAMES_PER_CODE:
        return np.repeat(model.vq.stats["mean"][None], T, axis=0).astype(np.float32), np.zeros(0, np.int64)
    f15, s15 = pool_pairs(f), pool_flag(s)
    if arch == "ar":
        if model.ar is None:
            raise ValueError("this checkpoint has no autoregressive model (a2m_ar.pt)")
        codes = model.ar.generate(f15, s15, causal=causal, temperature=temperature, top_p=top_p, seed=seed,
                                  repeat_penalty=repeat_penalty, stay_bias=stay_bias, stay_energy=stay_energy)
    else:
        if model.a2m is None:
            raise ValueError("this checkpoint has no feed-forward model (a2m.pt)")
        logits = model.a2m.logits(f15, s15, causal=causal)
        codes = sample_codes(logits, model.bigram_logp, temperature, bigram_weight, seed)
    z = model.vq.decode(codes)                                   # [2L, 14] standardised
    m = model.vq.denormalise(z)
    m = smooth_motion(m, RATE_HZ, smooth_hz)
    if len(m) < T:                                               # odd tail tick: hold the last frame
        m = np.concatenate([m, np.repeat(m[-1:], T - len(m), axis=0)], axis=0)
    return m[:T].astype(np.float32), codes


def generate(model: MotionModel, features: np.ndarray, speaking: np.ndarray, causal: bool = False,
             temperature: float = 0.8, bigram_weight: float = 0.5, seed: int = 0,
             smooth_hz: Optional[float] = DEFAULT_SMOOTH_HZ, arch: Optional[str] = None,
             top_p: float = 0.9, repeat_penalty: Optional[float] = None, stay_bias: Optional[float] = None,
             stay_energy: Optional[float] = None) -> HumanClip:
    """The public entry point: a valid ``HumanClip`` in the canonical space.
    ``arch`` and the AR sampling knobs default to the checkpoint's recorded values (``model_info.json``)."""
    rec = model.info.get("sampling", {})
    if arch is None:
        arch = model.info.get("default_arch") or ("ff" if model.a2m is not None else "ar")
    if repeat_penalty is None:
        repeat_penalty = float(rec.get("repeat_penalty", 0.0))
    if stay_bias is None:
        stay_bias = float(rec.get("stay_bias", 0.0))
    if stay_energy is None:
        stay_energy = float(rec.get("stay_energy", -0.3))
    m, codes = generate_motion(model, features, speaking, causal, temperature, bigram_weight, seed, smooth_hz,
                               arch=arch, top_p=top_p, repeat_penalty=repeat_penalty, stay_bias=stay_bias, stay_energy=stay_energy)
    clip = motion_to_clip(m, speaking, RATE_HZ, source="model", arch=arch, mode="listen" if causal else "talk",
                          seed=int(seed), temperature=float(temperature), bigram_weight=float(bigram_weight),
                          top_p=float(top_p), repeat_penalty=float(repeat_penalty), stay_bias=float(stay_bias),
                          stay_energy=float(stay_energy),
                          n_codes_sampled=int(len(codes)), distinct_codes=int(len(set(codes.tolist()))))
    return clip
