"""web/js/features.js must equal animacy/features.py; web/js/dsp.js filtfilt must equal scipy.

Both sides get the same synthetic 16 kHz waveform (harmonic voice bursts + noise,
seeded) and the outputs are diffed to 1e-4 after normalisation. Needs `node`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from animacy.features import N_FEATS, SR, audio_features, mel_filterbank

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARNESS = os.path.join(ROOT, "web", "dev", "features_parity.mjs")
NODE = shutil.which("node")


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    for s in np.arange(0.2, seconds - 0.3, 0.45):
        a, b = int(s * SR), int((s + 0.25) * SR)
        env[a:b] = np.hanning(b - a) * rng.uniform(0.4, 1.0)
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 8))
    wav = env * (0.6 * voice + 0.2 * rng.normal(0, 1, n)) + 0.005 * rng.normal(0, 1, n)
    return (0.8 * wav / np.abs(wav).max()).astype(np.float32)


def _run_node(job: dict) -> dict:
    res = subprocess.run([NODE, HARNESS], input=json.dumps(job), capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("seconds,seed", [(2.0, 0), (0.7, 3), (3.37, 11)])
def test_features_match(seconds: float, seed: int):
    wav = _speechlike(seconds, seed)
    ref = audio_features(wav, SR)
    js = np.asarray(_run_node({"wav": wav.tolist(), "n_ticks": None})["features"], dtype=np.float32)
    assert js.shape == ref.shape == (int(np.ceil(seconds * 30)), N_FEATS), (js.shape, ref.shape)
    d = np.abs(js - ref)
    assert d.max() < 1e-4, f"max |js - py| = {d.max():.2e} at {np.unravel_index(d.argmax(), d.shape)}"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_features_short_and_explicit_ticks():
    wav = _speechlike(0.05, 5)  # shorter than one window: zero-pad branch
    ref = audio_features(wav, SR, n_ticks=4)
    js = np.asarray(_run_node({"wav": wav.tolist(), "n_ticks": 4})["features"], dtype=np.float32)
    assert js.shape == ref.shape == (4, N_FEATS)
    assert np.abs(js - ref).max() < 1e-4


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_filtfilt_matches_scipy():
    from scipy.signal import butter, filtfilt

    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 90)) + 5 * np.sin(np.arange(90) / 3)
    for cutoff, rate, padlen in [(6.0, 30.0, 9), (0.3, 30.0, 9), (8.0, 30.0, 5)]:
        b, a = butter(2, min(cutoff / (0.5 * rate), 0.99))
        ref = filtfilt(b, a, x, padlen=padlen)
        out = _run_node({"wav": [0.0] * 800, "n_ticks": 2, "filt": {"x": x.tolist(), "cutoff_hz": cutoff, "rate_hz": rate, "padlen": padlen}})
        assert np.allclose(out["coeffs"]["b"], b, atol=1e-12) and np.allclose(out["coeffs"]["a"], a, atol=1e-12), (out["coeffs"], b, a)
        d = np.abs(np.asarray(out["filt"]) - ref).max()
        assert d < 1e-9, f"filtfilt cutoff={cutoff}: max diff {d:.2e}"


def test_mel_filterbank_shape():
    fb = mel_filterbank()
    assert fb.shape == (64, 257)
