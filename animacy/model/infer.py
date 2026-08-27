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


PITCH_IDX = MODEL_CHANNELS.index("head_pitch")
BASELINE_HZ = 0.3          # the same cutoff the pose channels were detrended with (data.DETREND_HZ)
QUIET_ENERGY = -0.3        # normalised log energy below this = silence (feature 64)
ENERGY_CHANNELS = ["head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "brow_l", "brow_r", "brow_furrow"]
ENERGY_IDX = [MODEL_CHANNELS.index(c) for c in ENERGY_CHANNELS]
ENERGY_FLOOR_CAP = 2.0     # the whole-utterance scale never exceeds this (and never goes below 1.0)


def motion_energy(motion: np.ndarray, stats: Optional[Dict] = None) -> float:
    """Per-utterance energy: RMS of the mean-removed motion over the 6 head channels + 3 brow
    channels, each standardised by the corpus std (so mm and degrees weigh alike). This is
    the quantity ``energy_floor`` is compared against."""
    m = np.asarray(motion, np.float64)[:, ENERGY_IDX]
    if len(m) == 0:
        return 0.0
    if stats is not None:
        m = m / np.asarray(stats["std"], np.float64)[ENERGY_IDX]
    m = m - m.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.square(m))))


def postprocess_motion(motion: np.ndarray, speaking: Optional[np.ndarray] = None, features: Optional[np.ndarray] = None,
                       settle_s: float = 0.0, pitch_floor: Optional[float] = None, amplitude=1.0,
                       rate_hz: float = RATE_HZ, energy_floor: Optional[float] = None,
                       energy_stats: Optional[Dict] = None) -> np.ndarray:
    """Generation-side options on a [T, 14] canonical-unit motion (applied to every source:
    model, AR, retrieval), in this order:

    * ``amplitude``: per-channel scale (scalar or [14]); the intent tier (1.45 excitement ... 0.9 thinking).
    * ``energy_floor``: if the utterance's ``motion_energy`` (standardised, mean-removed RMS over
      head + brows) is below this reference (the corpus's 60th percentile over 3 s windows, i.e.
      "a clearly moving human"), the WHOLE utterance is scaled by one scalar in [1, 2] to reach
      it. 0 / None disables.
    * ``pitch_floor``: the low-frequency (``BASELINE_HZ``) mean of ``head_pitch`` is never let
      below this many degrees - the baseline is LIFTED where it is low; nothing is flattened.
    * ``settle_s``: only after speech has ENDED (last ``speaking`` frame, else last frame with
      energy above ``QUIET_ENERGY``; never mid-utterance) the motion blends to neutral (0)
      over ``settle_s`` seconds and stays there, so a clip ends attentive instead of mid-gesture.
      If speech runs to the clip end nothing is settled.
    """
    m = np.array(motion, dtype=np.float32, copy=True)
    T, C = m.shape
    amp = np.broadcast_to(np.asarray(amplitude, np.float32), (C,))
    if np.any(amp != 1.0):
        m = m * amp
    if energy_floor and T > 1:
        e = motion_energy(m, energy_stats)
        if e > 1e-6 and e < energy_floor:
            m = m * float(min(ENERGY_FLOOR_CAP, max(1.0, energy_floor / e)))
    if pitch_floor is not None and T > 0:
        p = m[:, PITCH_IDX].astype(np.float64)
        if T >= 12:
            from scipy.signal import butter, filtfilt

            b, a = butter(2, BASELINE_HZ / (0.5 * rate_hz))
            base = filtfilt(b, a, p, padlen=min(9, T - 1))
        else:
            base = np.full(T, p.mean())
        m[:, PITCH_IDX] = (p + np.maximum(0.0, pitch_floor - base)).astype(np.float32)
    if settle_s and settle_s > 0 and T > 0:
        n = max(1, int(round(settle_s * rate_hz)))
        end = T
        if speaking is not None and len(speaking) == T and np.any(np.asarray(speaking) > 0):
            end = int(np.max(np.nonzero(np.asarray(speaking) > 0)[0])) + 1
        elif features is not None and len(features) == T and features.shape[1] > 64:
            idx = np.nonzero(np.asarray(features)[:, 64] > QUIET_ENERGY)[0]
            end = int(idx[-1]) + 1 if len(idx) else T
        if end < T:                                   # only after speech has ended, never mid-utterance
            w = np.zeros(T, np.float32)
            ramp = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[1:]
            k = min(n, T - end)
            w[end:end + k] = ramp[:k]
            w[end + k:] = 1.0
            m = m * (1.0 - w)[:, None]
    return m


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
                    stay_energy: float = -0.3, settle_s: float = 0.0, pitch_floor: Optional[float] = None,
                    amplitude=1.0, energy_floor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """[T, 66], [T] on the 30 Hz grid -> (motion [T, 14] raw units, codes [T//2]).
    ``arch`` = "ff" (per-step logits + bigram prior) or "ar" (autoregressive: temperature, top-p,
    repeat penalty, stay bias when the audio is quieter than ``stay_energy``). The last three
    arguments are the ``postprocess_motion`` options."""
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
    m = m[:T]
    if settle_s or pitch_floor is not None or energy_floor or np.any(np.asarray(amplitude) != 1.0):
        m = postprocess_motion(m, s, f, settle_s=settle_s, pitch_floor=pitch_floor, amplitude=amplitude,
                               energy_floor=energy_floor, energy_stats=model.vq.stats)
    return m.astype(np.float32), codes


def generate(model: MotionModel, features: np.ndarray, speaking: np.ndarray, causal: bool = False,
             temperature: float = 0.8, bigram_weight: float = 0.5, seed: int = 0,
             smooth_hz: Optional[float] = DEFAULT_SMOOTH_HZ, arch: Optional[str] = None,
             top_p: float = 0.9, repeat_penalty: Optional[float] = None, stay_bias: Optional[float] = None,
             stay_energy: Optional[float] = None, settle_s: Optional[float] = None,
             pitch_floor: Optional[float] = None, amplitude=None, intent=None,
             energy_floor: Optional[float] = None) -> HumanClip:
    """The public entry point: a valid ``HumanClip`` in the canonical space.
    ``arch``, the AR sampling knobs and the post-processing options default to the checkpoint's
    recorded values (``model_info.json`` -> ``sampling`` / ``postprocess``). ``intent`` (the
    utterance text, an ``intent.Intent`` or a tag name) sets the amplitude by the intent rule
    (``0.8 + 0.5 * arousal``) unless ``amplitude`` is given explicitly."""
    rec = model.info.get("sampling", {})
    pp = model.info.get("postprocess", {})
    intent_obj = None
    if intent is not None:
        from .intent import TAGS, Intent, analyse

        if isinstance(intent, Intent):
            intent_obj = intent
        elif isinstance(intent, str) and intent in TAGS:
            intent_obj = analyse("", override=intent)
        else:
            intent_obj = analyse(str(intent))
        if amplitude is None:
            amplitude = intent_obj.amplitude
    if arch is None:
        arch = model.info.get("default_arch") or ("ff" if model.a2m is not None else "ar")
    if repeat_penalty is None:
        repeat_penalty = float(rec.get("repeat_penalty", 0.0))
    if stay_bias is None:
        stay_bias = float(rec.get("stay_bias", 0.0))
    if stay_energy is None:
        stay_energy = float(rec.get("stay_energy", -0.3))
    if settle_s is None:
        settle_s = float(pp.get("settle_s", 0.0))
    if pitch_floor is None:
        pitch_floor = pp.get("pitch_floor", None)
    if amplitude is None:
        amplitude = pp.get("amplitude", 1.0)
    if energy_floor is None:
        energy_floor = pp.get("energy_floor", None)
    m, codes = generate_motion(model, features, speaking, causal, temperature, bigram_weight, seed, smooth_hz,
                               arch=arch, top_p=top_p, repeat_penalty=repeat_penalty, stay_bias=stay_bias, stay_energy=stay_energy,
                               settle_s=settle_s, pitch_floor=pitch_floor, amplitude=amplitude, energy_floor=energy_floor)
    clip = motion_to_clip(m, speaking, RATE_HZ, source="model", arch=arch, mode="listen" if causal else "talk",
                          seed=int(seed), temperature=float(temperature), bigram_weight=float(bigram_weight),
                          top_p=float(top_p), repeat_penalty=float(repeat_penalty), stay_bias=float(stay_bias),
                          stay_energy=float(stay_energy), settle_s=float(settle_s),
                          pitch_floor=(None if pitch_floor is None else float(pitch_floor)),
                          amplitude=(float(amplitude) if np.isscalar(amplitude) else [float(x) for x in amplitude]),
                          energy_floor=(None if not energy_floor else float(energy_floor)), energy=motion_energy(m, model.vq.stats),
                          n_codes_sampled=int(len(codes)), distinct_codes=int(len(set(codes.tolist()))))
    if intent_obj is not None:
        clip.meta["intent"] = {k: v for k, v in intent_obj.to_dict().items() if k != "hits"}
    return clip


def retrieve(index, features: np.ndarray, speaking: np.ndarray, model: Optional[MotionModel] = None,
             intent=None, settle_s: Optional[float] = None, pitch_floor: Optional[float] = None,
             amplitude=None, use_audio_arousal: bool = False, proto_weight: Optional[float] = None,
             energy_floor: Optional[float] = None, **meta) -> HumanClip:
    """The retrieval source with the same post-processing and intent handling as ``generate``:
    text intent -> arousal bonus + gesture-prototype bonus (``proto_weight``) in the index query,
    amplitude tier, energy floor. Without text the query is plain: the audio-only arousal proxy
    (``use_audio_arousal``) is off by default because it made retrieval stiller and less
    beat-aligned on both held-out speakers (v2a REPORT). ``proto_weight`` / ``energy_floor``
    default to the bundle's ``postprocess`` values (0 disables either)."""
    pp = (model.info.get("postprocess", {}) if model is not None else {})
    if settle_s is None:
        settle_s = float(pp.get("settle_s", 0.0))
    if pitch_floor is None:
        pitch_floor = pp.get("pitch_floor", None)
    if proto_weight is None:
        proto_weight = float(pp.get("proto_weight", 0.25))
    if energy_floor is None:
        energy_floor = pp.get("energy_floor", None)
    intent_obj, target, tag = None, None, None
    if intent is not None:
        from .intent import TAGS, Intent, analyse

        intent_obj = intent if isinstance(intent, Intent) else (analyse("", override=intent) if isinstance(intent, str) and intent in TAGS else analyse(str(intent)))
        target, tag = intent_obj.arousal, intent_obj.tag
        if amplitude is None:
            amplitude = intent_obj.amplitude
    if amplitude is None:
        amplitude = pp.get("amplitude", 1.0)
    f = np.asarray(features, np.float32)
    s = np.asarray(speaking, np.int64)
    m, ids = index.query(f, s, target_arousal=target, intent_tag=tag, use_audio_arousal=(target is None and use_audio_arousal),
                         proto_weight=proto_weight, return_ids=True)
    stats = model.vq.stats if model is not None else None
    m = postprocess_motion(m, s, f, settle_s=settle_s, pitch_floor=pitch_floor, amplitude=amplitude,
                           energy_floor=energy_floor, energy_stats=stats)
    proto_mean = None
    if tag is not None and index.proto is not None and tag in index.proto and len(ids):
        proto_mean = float(np.mean(index.proto[tag][np.asarray(ids)]))
    clip = motion_to_clip(m, s, RATE_HZ, source="retrieval", settle_s=float(settle_s),
                          pitch_floor=(None if pitch_floor is None else float(pitch_floor)),
                          amplitude=(float(amplitude) if np.isscalar(amplitude) else [float(x) for x in amplitude]),
                          proto_weight=float(proto_weight), proto_mean=proto_mean,
                          energy_floor=(None if not energy_floor else float(energy_floor)), energy=motion_energy(m, stats), **meta)
    if intent_obj is not None:
        clip.meta["intent"] = {k: v for k, v in intent_obj.to_dict().items() if k != "hits"}
    return clip
