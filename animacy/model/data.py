"""Clips -> training tensors for the motion model (``docs/MODEL.md``).

Everything the model ever sees is built here: which frames count (``face_valid``
in contiguous runs of at least one second), the 14 *model channels*, the audio
features on the same 30 Hz grid, the ``speaking`` flag, and the two windowings
(8-frame segments for the tokenizer, 2-second chunks for audio -> codes).
Splits are by clip, or by subject when ``meta.subject`` exists, never by frame.

``make_synthetic_clips`` fabricates correlated speech + motion in the real clip
layout so the whole pipeline is exercised before any capture exists.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import N_FEATS, SR as FEAT_SR, audio_features
from ..schema import BOUNDS, FACE_CHANNELS, HumanClip, MOTION_FILE, RATE_HZ, TORSO_CHANNELS, empty_frames

MODEL_CHANNELS: List[str] = [
    "head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z",
    "brow_l", "brow_r", "brow_furrow",
    "torso_lean_fwd", "torso_lean_side", "torso_yaw",
    "mouth_open", "smile",
]
N_MODEL = len(MODEL_CHANNELS)
FRAMES_PER_CODE = 2          # 30 Hz frames -> 15 Hz codes
SEGMENT = 8                  # frames per VQ training window
SEGMENT_STRIDE = 4
CHUNK_FRAMES = 60            # 2 s a2m training chunks
CHUNK_STRIDE = 15            # 0.5 s hop: overlapping chunks, the cheapest augmentation there is
CHUNK_STEPS = CHUNK_FRAMES // FRAMES_PER_CODE
MIN_RUN = 30                 # 1 s of continuous face tracking
NORM_CLIP = 4.0              # standardised motion is clipped to +-4 sd (decoder range)
FEATURE_CONTRACT = "animacy.features.v1"
# Pose channels are modelled as the residual above DETREND_HZ: where a person *holds*
# their head is not something speech predicts, the nods / tilts / glances on top of it
# are. The robot therefore stays centred and the gaze overlay owns the baseline.
DETREND_HZ = 0.3
POSE_CHANNELS = ["head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z",
                 "torso_lean_fwd", "torso_lean_side", "torso_yaw"]
# Training motion is low-passed at SMOOTH_HZ inside each run: landmark jitter is not motion,
# and a 512 codebook will happily spend its entries on it (real-data perplexity 455 before this).
SMOOTH_HZ = 8.0

_FACE_IDX = [i for i, c in enumerate(MODEL_CHANNELS) if c in FACE_CHANNELS]
_POSE_IDX = [MODEL_CHANNELS.index(c) for c in POSE_CHANNELS]


def detrend_runs(motion: np.ndarray, runs: Sequence[Tuple[int, int]], cutoff_hz: float = DETREND_HZ,
                 rate_hz: float = RATE_HZ) -> np.ndarray:
    """Subtract a zero-phase low-pass (``cutoff_hz``) of the pose channels inside each run."""
    from scipy.signal import butter, filtfilt

    out = np.array(motion, dtype=np.float32, copy=True)
    if not cutoff_hz:
        return out
    b, a = butter(2, cutoff_hz / (0.5 * rate_hz))
    for s, e in runs:
        seg = out[s:e, _POSE_IDX].astype(np.float64)
        if len(seg) < 12:
            base = seg.mean(axis=0, keepdims=True)
        else:
            base = filtfilt(b, a, seg, axis=0, padlen=min(9, len(seg) - 1))
        out[s:e, _POSE_IDX] = (seg - base).astype(np.float32)
    return out


def smooth_runs(motion: np.ndarray, runs: Sequence[Tuple[int, int]], cutoff_hz: float = SMOOTH_HZ,
                rate_hz: float = RATE_HZ) -> np.ndarray:
    """Zero-phase low-pass of every channel inside each run; bounded channels stay in bounds."""
    from scipy.signal import butter, filtfilt

    out = np.array(motion, dtype=np.float32, copy=True)
    if not cutoff_hz:
        return out
    b, a = butter(2, min(cutoff_hz / (0.5 * rate_hz), 0.99))
    lo = np.array([BOUNDS[c][0] for c in MODEL_CHANNELS], np.float32)
    hi = np.array([BOUNDS[c][1] for c in MODEL_CHANNELS], np.float32)
    for s, e in runs:
        if e - s < 12:
            continue
        seg = filtfilt(b, a, out[s:e].astype(np.float64), axis=0, padlen=min(9, e - s - 1))
        out[s:e] = np.clip(seg, lo, hi).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# clip -> arrays
# ---------------------------------------------------------------------------
@dataclass
class ClipData:
    """One clip on the 30 Hz grid. ``runs`` lists the frame ranges that count."""

    name: str
    subject: str
    source: str
    features: np.ndarray        # [T, 66] float32 (zeros when the clip has no audio)
    motion: np.ndarray          # [T, 14] float32, raw canonical units, NaN -> 0
    speaking: np.ndarray        # [T] int64
    runs: List[Tuple[int, int]]
    has_audio: bool = True

    @property
    def n_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def n_valid(self) -> int:
        return int(sum(b - a for a, b in self.runs))

    @property
    def group(self) -> str:
        return self.subject if self.subject else f"clip:{self.name}"


def contiguous_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def _to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == FEAT_SR:
        return np.asarray(wav, dtype=np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(int(sr), FEAT_SR)
    return resample_poly(np.asarray(wav, dtype=np.float64), FEAT_SR // g, int(sr) // g).astype(np.float32)


def clip_to_data(clip: HumanClip, name: str) -> Optional[ClipData]:
    f = clip.frames
    if abs(clip.rate_hz - RATE_HZ) > 1e-6:
        print(f"  skip {name}: rate_hz={clip.rate_hz} (model grid is {RATE_HZ})")
        return None
    n = len(f)
    if n < MIN_RUN:
        return None
    motion = f[MODEL_CHANNELS].to_numpy(dtype=np.float32)
    valid = np.nan_to_num(f["face_valid"].to_numpy(dtype=np.float32)) > 0
    valid &= np.all(np.isfinite(motion[:, _FACE_IDX]), axis=1)
    motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)   # torso NaN -> neutral
    speaking = (np.nan_to_num(f["speaking"].to_numpy(dtype=np.float32)) > 0).astype(np.int64)
    runs = contiguous_runs(valid, MIN_RUN)
    motion = smooth_runs(detrend_runs(motion, runs), runs)
    has_audio = clip.audio is not None and len(clip.audio) > 0
    if has_audio:
        features = audio_features(_to_16k(clip.audio, clip.sr), FEAT_SR, n_ticks=n)
    else:
        features = np.zeros((n, N_FEATS), dtype=np.float32)
    return ClipData(
        name=name,
        subject=str(clip.meta.get("subject") or ""),
        source=str(clip.meta.get("source") or ""),
        features=features.astype(np.float32),
        motion=motion,
        speaking=speaking,
        runs=runs,
        has_audio=has_audio,
    )


def list_clip_dirs(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    out = []
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, MOTION_FILE)):
            out.append(p)
    return out


def load_clips(data_dir: str, verbose: bool = True) -> List[ClipData]:
    clips: List[ClipData] = []
    for p in list_clip_dirs(data_dir):
        name = os.path.basename(p)
        try:
            hc = HumanClip.load(p, audio=True)
        except Exception as e:  # a half-written clip from a concurrent capture must not kill training
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue
        cd = clip_to_data(hc, name)
        if cd is None:
            continue
        clips.append(cd)
        if verbose:
            print(f"  {name}: {cd.n_frames} frames, {len(cd.runs)} runs, {cd.n_valid / RATE_HZ:.1f}s valid, "
                  f"audio={'yes' if cd.has_audio else 'NO'}, subject={cd.subject or '-'}, source={cd.source or '-'}")
    return clips


def summarise(clips: Sequence[ClipData]) -> Dict:
    sources: Dict[str, int] = {}
    for c in clips:
        sources[c.source or "unknown"] = sources.get(c.source or "unknown", 0) + 1
    return {
        "n_clips": len(clips),
        "n_with_audio": int(sum(c.has_audio for c in clips)),
        "n_subjects": len({c.subject for c in clips if c.subject}),
        "total_minutes": round(sum(c.n_frames for c in clips) / RATE_HZ / 60.0, 2),
        "valid_minutes": round(sum(c.n_valid for c in clips) / RATE_HZ / 60.0, 2),
        "valid_minutes_with_audio": round(sum(c.n_valid for c in clips if c.has_audio) / RATE_HZ / 60.0, 2),
        "sources": sources,
        "clips": [c.name for c in clips],
    }


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def split_clips(clips: Sequence[ClipData], val_frac: float = 0.2, seed: int = 0,
                holdout: Optional[Sequence[str]] = None):
    """Hold out whole groups (subject if known, else clip). ``holdout`` names clips
    or subjects to hold out explicitly; otherwise, deterministically the group
    whose valid duration is nearest ``val_frac`` of the corpus (a random pick on
    six speakers once held out a 24 s outlier), adding groups only while the
    hold-out is under half the target. With a single group the only option is a
    time split inside each run: leaky, and flagged."""
    groups: Dict[str, List[ClipData]] = {}
    for c in clips:
        groups.setdefault(c.group, []).append(c)
    keys = sorted(groups)
    if holdout:
        wanted = set(holdout)
        val_keys = sorted({c.group for c in clips if c.name in wanted or c.subject in wanted or c.group in wanted})
        missing = sorted(w for w in wanted if not any(w in (c.name, c.subject, c.group) for c in clips))
        if missing:
            raise ValueError(f"--holdout names not found among clips/subjects: {missing}")
        if not val_keys or len(val_keys) >= len(keys):
            raise ValueError("holdout must leave at least one group for training")
        train = [c for k in keys if k not in val_keys for c in groups[k]]
        val = [c for k in keys if k in val_keys for c in groups[k]]
        info = {"mode": "explicit", "n_groups": len(keys), "held_out_groups": val_keys, "leaky": False,
                "held_out_clips": [c.name for c in val],
                "held_out_valid_seconds": round(sum(c.n_valid for c in val) / RATE_HZ, 1), "rule": "--holdout"}
        return train, val, info
    if len(keys) >= 2:
        sizes = {k: sum(c.n_valid for c in groups[k]) for k in keys}
        target = val_frac * sum(sizes.values())
        val_keys = []
        while True:
            rest = [k for k in keys if k not in val_keys]
            deficit = target - sum(sizes[k] for k in val_keys)
            rest.sort(key=lambda k: (abs(sizes[k] - deficit), k))
            val_keys.append(rest[0])
            if sum(sizes[k] for k in val_keys) >= 0.5 * target or len(val_keys) >= len(keys) - 1:
                break
        train = [c for k in keys if k not in val_keys for c in groups[k]]
        val = [c for k in keys if k in val_keys for c in groups[k]]
        mode = "subject" if any(c.subject for c in clips) else "clip"
        info = {"mode": mode, "n_groups": len(keys), "held_out_groups": sorted(val_keys), "leaky": False,
                "held_out_clips": [c.name for c in val],
                "held_out_valid_seconds": round(sum(sizes[k] for k in val_keys) / RATE_HZ, 1),
                "rule": f"group(s) nearest {val_frac:.0%} of valid frames"}
        return train, val, info
    train, val = [], []
    for c in clips:
        tr, va = [], []
        for a, b in c.runs:
            cut = a + int((b - a) * (1.0 - val_frac))
            if cut - a >= MIN_RUN and b - cut >= MIN_RUN:
                tr.append((a, cut))
                va.append((cut, b))
            else:
                tr.append((a, b))
        train.append(replace(c, runs=tr))
        if va:
            val.append(replace(c, runs=va))
    info = {"mode": "time-within-run (single group: NOT a speaker hold-out)", "n_groups": len(keys),
            "held_out_groups": [], "leaky": True}
    return train, val, info


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def compute_stats(clips: Sequence[ClipData]) -> Dict[str, np.ndarray]:
    xs = [c.motion[a:b] for c in clips for a, b in c.runs]
    if not xs:
        raise ValueError("no valid frames to compute statistics from")
    x = np.concatenate(xs, axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-3)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalise(motion: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    z = (np.asarray(motion, dtype=np.float32) - stats["mean"]) / stats["std"]
    return np.clip(z, -NORM_CLIP, NORM_CLIP).astype(np.float32)


def denormalise(z: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(z, dtype=np.float32) * stats["std"] + stats["mean"]).astype(np.float32)


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def pool_pairs(x: np.ndarray) -> np.ndarray:
    """30 Hz -> 15 Hz by averaging consecutive pairs (an odd tail tick is dropped)."""
    x = np.asarray(x, dtype=np.float32)
    n = (len(x) // FRAMES_PER_CODE) * FRAMES_PER_CODE
    return x[:n].reshape(n // FRAMES_PER_CODE, FRAMES_PER_CODE, *x.shape[1:]).mean(axis=1)


def pool_flag(x: np.ndarray) -> np.ndarray:
    """30 Hz flag -> 15 Hz flag: any of the pair."""
    x = np.asarray(x)
    n = (len(x) // FRAMES_PER_CODE) * FRAMES_PER_CODE
    return x[:n].reshape(n // FRAMES_PER_CODE, FRAMES_PER_CODE).max(axis=1).astype(np.int64)


def vq_segments(clips: Sequence[ClipData], stats: Dict[str, np.ndarray],
                segment: int = SEGMENT, stride: int = SEGMENT_STRIDE) -> np.ndarray:
    """[N, segment, 14] standardised windows, only from inside contiguous runs."""
    out = []
    for c in clips:
        for a, b in c.runs:
            z = normalise(c.motion[a:b], stats)
            for s in range(0, len(z) - segment + 1, stride):
                out.append(z[s:s + segment])
    if not out:
        return np.zeros((0, segment, N_MODEL), dtype=np.float32)
    return np.stack(out).astype(np.float32)


def a2m_chunks(clips: Sequence[ClipData], stats: Dict[str, np.ndarray],
               encode: Callable[[np.ndarray], np.ndarray],
               chunk: int = CHUNK_FRAMES, stride: int = CHUNK_STRIDE) -> Dict[str, np.ndarray]:
    """2 s chunks for audio -> codes. Short tails (>= 1 s) are kept and padded;
    ``mask`` marks real steps. ``encode`` maps standardised frames [n,14] -> codes [n/2]."""
    steps = chunk // FRAMES_PER_CODE
    F, S, C, M = [], [], [], []
    for c in clips:
        if not c.has_audio:
            continue
        for a, b in c.runs:
            for s in range(a, b, stride):
                e = min(s + chunk, b)
                n = ((e - s) // FRAMES_PER_CODE) * FRAMES_PER_CODE
                if n < MIN_RUN:
                    continue
                feats = pool_pairs(c.features[s:s + n])
                spk = pool_flag(c.speaking[s:s + n])
                codes = np.asarray(encode(normalise(c.motion[s:s + n], stats)), dtype=np.int64)
                L = len(codes)
                assert L == len(feats) == len(spk), (L, len(feats), len(spk))
                pf = np.zeros((steps, N_FEATS), np.float32)
                ps = np.zeros(steps, np.int64)
                pc = np.zeros(steps, np.int64)
                pm = np.zeros(steps, bool)
                pf[:L], ps[:L], pc[:L], pm[:L] = feats, spk, codes, True
                F.append(pf), S.append(ps), C.append(pc), M.append(pm)
                if e == b:
                    break
    if not F:
        z = np.zeros((0, steps), np.int64)
        return {"features": np.zeros((0, steps, N_FEATS), np.float32), "speaking": z, "codes": z, "mask": z.astype(bool)}
    return {"features": np.stack(F), "speaking": np.stack(S), "codes": np.stack(C), "mask": np.stack(M)}


def run_code_sequences(clips: Sequence[ClipData], stats: Dict[str, np.ndarray],
                       encode: Callable[[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """One code sequence per contiguous run (for the bigram prior / usage stats)."""
    out = []
    for c in clips:
        for a, b in c.runs:
            n = ((b - a) // FRAMES_PER_CODE) * FRAMES_PER_CODE
            if n >= FRAMES_PER_CODE:
                out.append(np.asarray(encode(normalise(c.motion[a:a + n], stats)), dtype=np.int64))
    return out


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------
def _ou(rng: np.random.Generator, n: int, sd: float, tau_s: float, rate: float = RATE_HZ) -> np.ndarray:
    """Ornstein-Uhlenbeck noise with stationary sd and correlation time tau."""
    a = math.exp(-1.0 / (tau_s * rate))
    noise = rng.normal(0.0, sd * math.sqrt(1.0 - a * a), n)
    x = np.zeros(n)
    v = rng.normal(0.0, sd)
    for i in range(n):
        v = a * v + noise[i]
        x[i] = v
    return x


def _add_bump(x: np.ndarray, center: int, width: int, amp: float) -> None:
    lo, hi = max(0, center - width // 2), min(len(x), center + width // 2 + 1)
    if hi <= lo:
        return
    k = np.arange(lo, hi) - (center - width // 2)
    x[lo:hi] += amp * np.hanning(width + 1)[k]


def _add_nod(x: np.ndarray, t0: int, amp: float, freq: float = 2.2, decay_s: float = 0.25, rate: float = RATE_HZ) -> None:
    n = int(1.2 * rate)
    tau = np.arange(n) / rate
    y = amp * np.sin(2 * np.pi * freq * tau) * np.exp(-tau / decay_s)
    hi = min(len(x), t0 + n)
    if hi > t0 >= 0:
        x[t0:hi] += y[:hi - t0]


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def make_synthetic_clip(seed: int, seconds: float = 20.0, subject: str = "synth0", listening: bool = False,
                        face_gap: bool = False, torso_missing: bool = False, rate: float = RATE_HZ):
    """Plausible correlated speech + motion: syllable bursts drive mouth_open,
    phrase onsets trigger nods / brow raises / lean-ins, slow OU drift elsewhere."""
    rng = np.random.default_rng(seed)
    T = int(round(seconds * rate))
    speak = np.zeros(T, dtype=bool)
    phrases: List[Tuple[int, int]] = []
    t = int(rng.uniform(0.3, 1.5) * rate)
    while t < T - int(0.5 * rate):
        dur = int(rng.uniform(0.6, 1.5) * rate) if listening else int(rng.uniform(1.5, 4.0) * rate)
        e = min(T, t + dur)
        speak[t:e] = True
        phrases.append((t, e))
        gap = rng.uniform(2.0, 5.0) if listening else rng.uniform(0.5, 2.0)
        t = e + int(gap * rate)

    # --- audio envelope at 30 Hz, syllable train inside phrases
    env = np.zeros(T)
    for s, e in phrases:
        loud = rng.uniform(0.5, 1.0)
        syl_rate = rng.uniform(3.5, 5.5)
        ts = s + 2.0
        while ts < e - 1:
            _add_bump(env, int(ts), int(0.18 * rate), loud * rng.uniform(0.4, 1.0))
            ts += rate / syl_rate * rng.uniform(0.8, 1.2)
    env = np.clip(env, 0.0, 1.5)

    # --- waveform: envelope x (harmonic voice + noise), plus room noise
    sr = FEAT_SR
    n_samp = int(round(seconds * sr))
    ts_s = np.arange(n_samp) / sr
    env_s = np.interp(ts_s, np.arange(T) / rate, env)
    f0 = 110.0 + 30.0 * (sum(ord(ch) for ch in subject) % 5) + 4.0 * np.sin(2 * np.pi * 5.0 * ts_s)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    harm = sum(np.sin(k * phase) / k for k in range(1, 6))
    wav = env_s * (0.7 * harm / 2.0 + 0.3 * rng.normal(0, 1, n_samp)) + 0.01 * rng.normal(0, 1, n_samp)
    wav = (0.8 * wav / (np.abs(wav).max() + 1e-6)).astype(np.float32)

    # --- motion
    energy = _smooth(env, 3)
    mouth = np.clip(0.9 * energy / (energy.max() + 1e-6) + 0.03 * np.abs(rng.normal(0, 1, T)), 0, 1)
    pitch = _ou(rng, T, 2.0, 2.0)
    yaw = _ou(rng, T, 6.0, 3.0)
    brow = np.zeros(T)
    furrow = np.zeros(T)
    lean_fwd = _ou(rng, T, 2.5, 6.0)
    for s, e in phrases:
        if rng.random() < 0.8:
            _add_nod(pitch, s, rng.uniform(-9, -3))
        tt = s + int(rng.uniform(0.6, 1.0) * rate)
        while tt < e - int(0.3 * rate):
            if rng.random() < 0.4:
                _add_nod(pitch, tt, rng.uniform(-5, -1.5), freq=rng.uniform(1.8, 2.8))
            tt += int(rng.uniform(0.6, 1.0) * rate)
        if rng.random() < 0.5:
            _add_nod(pitch, max(s, e - 5), rng.uniform(-6, -2))
        if rng.random() < 0.3:
            amp = rng.uniform(-15, 15)
            _add_bump(yaw, s + int(0.5 * rate), int(rng.uniform(0.8, 2.0) * rate), amp)
        if rng.random() < 0.45:
            _add_bump(brow, s + int(0.1 * rate), int(0.5 * rate), rng.uniform(0.3, 0.8))
        if rng.random() < 0.1:
            _add_bump(furrow, s + int(0.3 * rate), int(0.6 * rate), rng.uniform(0.2, 0.5))
        if rng.random() < 0.3 and (e - s) > 2 * rate:
            _add_bump(lean_fwd, s + int(1.0 * rate), int(2.0 * rate), rng.uniform(2, 4))
    # listener back-channel nods
    for tt in range(0, T):
        if not speak[tt] and rng.random() < 0.4 / rate:
            _add_nod(pitch, tt, rng.uniform(-4, -1))
    roll = _ou(rng, T, 2.0, 2.0) + 0.15 * yaw
    lean_side = _ou(rng, T, 1.5, 6.0)
    torso_yaw = 0.3 * yaw + _ou(rng, T, 2.0, 5.0)
    head_x = 1.2 * lean_fwd + _ou(rng, T, 4.0, 3.0)
    head_y = -0.35 * yaw + _ou(rng, T, 3.0, 3.0)
    head_z = 0.5 * pitch + _ou(rng, T, 3.0, 3.0)
    smile = np.clip(0.15 + _ou(rng, T, 0.12, 6.0), 0, 1)
    jit = lambda sd: rng.normal(0, sd, T)  # tracking jitter

    frames = empty_frames(T)
    frames["head_yaw"] = yaw + jit(0.25)
    frames["head_pitch"] = pitch + jit(0.25)
    frames["head_roll"] = roll + jit(0.25)
    frames["head_x"] = head_x + jit(0.5)
    frames["head_y"] = head_y + jit(0.5)
    frames["head_z"] = head_z + jit(0.5)
    frames["brow_l"] = np.clip(brow * (1.0 + 0.1 * rng.normal()) + 0.01 * np.abs(jit(1)), 0, 1)
    frames["brow_r"] = np.clip(brow * (1.0 + 0.1 * rng.normal()) + 0.01 * np.abs(jit(1)), 0, 1)
    frames["brow_furrow"] = np.clip(furrow + 0.01 * np.abs(jit(1)), 0, 1)
    frames["mouth_open"] = mouth
    frames["smile"] = smile
    frames["torso_lean_fwd"] = lean_fwd + jit(0.2)
    frames["torso_lean_side"] = lean_side + jit(0.2)
    frames["torso_yaw"] = torso_yaw + jit(0.2)
    for c in MODEL_CHANNELS:
        lo, hi = BOUNDS[c]
        frames[c] = np.clip(frames[c].to_numpy(), lo, hi)
    frames["speaking"] = speak.astype(np.float32)
    frames["face_valid"] = 1.0
    frames["arm_valid"] = 0.0
    if face_gap:
        g0 = int(rng.uniform(0.2, 0.7) * T)
        g1 = min(T, g0 + int(rng.uniform(0.4, 1.0) * rate))
        frames.loc[g0:g1 - 1, "face_valid"] = 0.0
        for c in FACE_CHANNELS:
            frames.loc[g0:g1 - 1, c] = np.nan
    if torso_missing:
        for c in TORSO_CHANNELS:
            frames[c] = np.nan
    return frames, wav


def make_synthetic_clips(out_dir: str, n_clips: int = 8, seconds: float = 20.0, seed: int = 0,
                         n_subjects: int = 3) -> List[str]:
    """Write ``n_clips`` synthetic clips in the real layout; returns their dirs."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n_clips):
        subject = f"synth{i % n_subjects}"
        frames, wav = make_synthetic_clip(seed * 1000 + i, seconds, subject=subject,
                                          listening=(i % 4 == 3), face_gap=(i % 2 == 0), torso_missing=(i % 4 == 1))
        clip = HumanClip.from_frames(frames, source="synthetic", subject=subject, license="self",
                                     rate_hz=RATE_HZ, arm="none", notes="fabricated by animacy.model.data.make_synthetic_clips")
        clip.audio = wav
        clip.sr = FEAT_SR
        errs = clip.validate()
        if errs:
            raise RuntimeError(f"synthetic clip {i} invalid: {errs}")
        paths.append(clip.save(os.path.join(out_dir, f"synthetic_{i:03d}")))
    return paths
