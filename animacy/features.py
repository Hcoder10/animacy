"""Audio features for the motion model — the contract mirrored by ``web/js/features.js``.

16 kHz mono float32 in → ``[T, 66]`` float32 out on the 30 Hz motion grid:
64 log-mel bands (win 25 ms, hop 10 ms, averaged per 33.3 ms tick) + log energy
+ delta log energy, then per-utterance mean/variance normalisation. Everything
is plain numpy so the JS port is line-for-line.
"""
from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 512
WIN = 400            # 25 ms
HOP = 160            # 10 ms → 100 Hz
N_MELS = 64
FMIN = 50.0
FMAX = 7600.0
RATE_HZ = 30.0
N_FEATS = N_MELS + 2
EPS = 1e-6


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """[n_mels, n_fft//2+1] triangular filters (HTK mel, Slaney-style area norm off)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    edges = mel_to_hz(mels)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(c - lo, EPS)
        down = (hi - freqs) / max(hi - c, EPS)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)
    return fb.astype(np.float32)


_FB = None


def log_mel_100hz(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """[N, 64] log-mel frames at 100 Hz (frame k centred at k*10 ms)."""
    global _FB
    if sr != SR:
        raise ValueError(f"expected {SR} Hz audio, got {sr}")
    if _FB is None:
        _FB = mel_filterbank()
    x = np.asarray(wav, dtype=np.float32)
    pad = WIN // 2
    x = np.pad(x, (pad, pad), mode="reflect") if len(x) > pad else np.pad(x, (pad, pad + WIN))
    n_frames = 1 + (len(x) - WIN) // HOP
    if n_frames <= 0:
        return np.zeros((0, N_MELS), dtype=np.float32)
    window = np.hanning(WIN + 1)[:-1].astype(np.float32)  # periodic Hann, same as JS
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * window
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = spec @ _FB.T
    return np.log(mel + EPS).astype(np.float32)


def to_motion_grid(feats_100hz: np.ndarray, n_ticks: int, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Average 100 Hz frames into each 1/rate_hz tick (tick i spans [i, i+1)/rate)."""
    out = np.zeros((n_ticks, feats_100hz.shape[1]), dtype=np.float32)
    for i in range(n_ticks):
        a = int(round(i * 100.0 / rate_hz))
        b = int(round((i + 1) * 100.0 / rate_hz))
        seg = feats_100hz[a:b] if b > a else feats_100hz[a:a + 1]
        if len(seg):
            out[i] = seg.mean(axis=0)
        elif i:
            out[i] = out[i - 1]
    return out


def energy_100hz(wav: np.ndarray) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    n_frames = max(1, 1 + (len(x) - WIN) // HOP) if len(x) >= WIN else 1
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, max(len(x) - 1, 0))
    fr = x[idx] if len(x) else np.zeros((1, WIN), np.float32)
    return np.log(np.sqrt((fr ** 2).mean(axis=1)) + EPS).astype(np.float32)


def normalise(feats: np.ndarray) -> np.ndarray:
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-3
    return ((feats - mu) / sd).astype(np.float32)


def audio_features(wav: np.ndarray, sr: int = SR, n_ticks: int | None = None, rate_hz: float = RATE_HZ) -> np.ndarray:
    """The full contract: ``[n_ticks, 66]`` normalised features on the motion grid."""
    if n_ticks is None:
        n_ticks = int(np.ceil(len(wav) / sr * rate_hz))
    mel = to_motion_grid(log_mel_100hz(wav, sr), n_ticks, rate_hz)
    en = to_motion_grid(energy_100hz(wav)[:, None], n_ticks, rate_hz)
    den = np.diff(en, axis=0, prepend=en[:1])
    return normalise(np.concatenate([mel, en, den], axis=1))
