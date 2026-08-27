"""ONNX export for the browser (ONNX Runtime Web) + the ``web/models/`` bundle.

* ``a2m.onnx``: features [1, L, 66] float32, speaking [1, L] int64,
  causal [1] int64 (0 = talk / non-causal, 1 = listen / causal) -> logits [1, L, 512].
* ``vq_decoder.onnx``: codes [1, L] int64 -> motion [1, 2L, 14] float32 in
  canonical units (de-standardised inside the graph).
* ``model.json``: channel list, stats, sampling defaults, the feature contract,
  and where the bigram log-probabilities live (``bigram.bin``, float16 [512, 512]).
* ``retrieval.json`` + ``retrieval.bin``: the motion-matching index.

Every graph is checked against torch with onnxruntime at two sequence lengths
(dynamic L) and the max abs difference is reported; ``verify`` fails loudly
above ``tol``.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..features import N_FEATS
from .data import FEATURE_CONTRACT, FRAMES_PER_CODE, MODEL_CHANNELS, NORM_CLIP
from .infer import DEFAULT_SMOOTH_HZ, MotionModel
from .retrieval import RetrievalIndex

OPSET = 17


class _A2MWrapper(nn.Module):
    def __init__(self, a2m: nn.Module) -> None:
        super().__init__()
        self.a2m = a2m

    def forward(self, features: torch.Tensor, speaking: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        return self.a2m(features, speaking, causal)


class _DecoderWrapper(nn.Module):
    def __init__(self, vq: nn.Module) -> None:
        super().__init__()
        self.codebook = nn.Parameter(vq.quantizer.codebook.detach().clone(), requires_grad=False)
        self.decoder = vq.decoder
        self.mean = nn.Parameter(vq.mean.detach().clone(), requires_grad=False)
        self.std = nn.Parameter(vq.std.detach().clone(), requires_grad=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        z = self.codebook[codes].permute(0, 2, 1)            # [1, dim, L]
        return self.decoder(z) * self.std + self.mean        # [1, 2L, 14]


def _export(module: nn.Module, args, path: str, input_names, output_names, dynamic_axes) -> str:
    module = module.eval()
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dynamo in (False, True):
            try:
                kw = dict(input_names=input_names, output_names=output_names, opset_version=OPSET, dynamic_axes=dynamic_axes)
                if dynamo:
                    kw["dynamo"] = True
                else:
                    kw["dynamo"] = False
                torch.onnx.export(module, args, path, **kw)
                return "dynamo" if dynamo else "torchscript"
            except Exception as e:  # noqa: BLE001 - try the other exporter
                errors.append(f"{'dynamo' if dynamo else 'torchscript'}: {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("ONNX export failed:\n  " + "\n  ".join(errors))


def _ort_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def export_a2m(model: MotionModel, path: str, verify_lengths=(30, 97), tol: float = 1e-4) -> Dict:
    a2m = model.a2m.eval().cpu()
    wrapper = _A2MWrapper(a2m).eval()
    L0 = 30
    feats = torch.randn(1, L0, N_FEATS)
    spk = torch.randint(0, 2, (1, L0), dtype=torch.int64)
    causal = torch.tensor([0], dtype=torch.int64)
    exporter = _export(wrapper, (feats, spk, causal), path, ["features", "speaking", "causal"], ["logits"],
                       {"features": {1: "L"}, "speaking": {1: "L"}, "logits": {1: "L"}})
    sess = _ort_session(path)
    worst = 0.0
    rng = np.random.default_rng(0)
    for L in verify_lengths:
        f = rng.normal(size=(1, L, N_FEATS)).astype(np.float32)
        s = rng.integers(0, 2, size=(1, L)).astype(np.int64)
        for c in (0, 1):
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(f), torch.from_numpy(s), torch.tensor([c], dtype=torch.int64)).numpy()
            got = sess.run(None, {"features": f, "speaking": s, "causal": np.array([c], np.int64)})[0]
            assert got.shape == ref.shape, (got.shape, ref.shape)
            worst = max(worst, float(np.abs(got - ref).max()))
    ok = worst < tol
    return {"path": path, "bytes": os.path.getsize(path), "exporter": exporter, "max_abs_diff": worst, "ok": ok,
            "verify_lengths": list(verify_lengths), "tol": tol}


def export_vq_decoder(model: MotionModel, path: str, verify_lengths=(15, 61), tol: float = 1e-4) -> Dict:
    vq = model.vq.eval().cpu()
    wrapper = _DecoderWrapper(vq).eval()
    codes = torch.randint(0, vq.n_codes, (1, 15), dtype=torch.int64)
    exporter = _export(wrapper, (codes,), path, ["codes"], ["motion"], {"codes": {1: "L"}, "motion": {1: "T"}})
    sess = _ort_session(path)
    worst = 0.0
    rng = np.random.default_rng(0)
    for L in verify_lengths:
        c = rng.integers(0, vq.n_codes, size=(1, L)).astype(np.int64)
        with torch.no_grad():
            ref = wrapper(torch.from_numpy(c)).numpy()
        got = sess.run(None, {"codes": c})[0]
        assert got.shape == (1, FRAMES_PER_CODE * L, len(MODEL_CHANNELS)), got.shape
        # the graph must also agree with the python decode path (denormalised)
        py = vq.denormalise(vq.decode(c[0]))
        worst = max(worst, float(np.abs(got - ref).max()), float(np.abs(got[0] - py).max()))
    ok = worst < tol
    return {"path": path, "bytes": os.path.getsize(path), "exporter": exporter, "max_abs_diff": worst, "ok": ok,
            "verify_lengths": list(verify_lengths), "tol": tol}


def export_bundle(model: MotionModel, index: Optional[RetrievalIndex], out_dir: str, metrics: Optional[Dict] = None,
                  tol: float = 1e-4) -> Dict:
    """Write ``a2m.onnx``, ``vq_decoder.onnx``, ``bigram.bin``, ``model.json`` (+ retrieval index)."""
    os.makedirs(out_dir, exist_ok=True)
    report: Dict = {}
    report["a2m"] = export_a2m(model, os.path.join(out_dir, "a2m.onnx"), tol=tol)
    report["vq_decoder"] = export_vq_decoder(model, os.path.join(out_dir, "vq_decoder.onnx"), tol=tol)

    bigram16 = np.asarray(model.bigram_logp, np.float32).astype(np.float16)
    bigram_path = os.path.join(out_dir, "bigram.bin")
    with open(bigram_path, "wb") as fh:
        fh.write(bigram16.tobytes())
    report["bigram"] = {"path": bigram_path, "bytes": os.path.getsize(bigram_path)}

    stats = model.vq.stats
    verdict = (metrics or {}).get("verdict", {})
    model_json = {
        "schema": "animacy.model.v1",
        "feature_contract": FEATURE_CONTRACT,
        "rate_hz": 30,
        "feature_rate_hz": 30 // FRAMES_PER_CODE,
        "frames_per_code": FRAMES_PER_CODE,
        "n_feats": N_FEATS,
        "n_codes": model.n_codes,
        "channels": list(MODEL_CHANNELS),
        "stats": {"mean": [float(x) for x in stats["mean"]], "std": [float(x) for x in stats["std"]],
                  "norm_clip": NORM_CLIP},
        "a2m": {
            "file": "a2m.onnx",
            "inputs": {"features": "float32 [1, L, 66] (30 Hz features averaged in pairs -> 15 Hz)",
                       "speaking": "int64 [1, L] (any of the pair)",
                       "causal": "int64 [1]: 0 = talk (non-causal), 1 = listen (causal)"},
            "outputs": {"logits": "float32 [1, L, 512]"},
            "bytes": report["a2m"]["bytes"],
        },
        "vq_decoder": {
            "file": "vq_decoder.onnx",
            "inputs": {"codes": "int64 [1, L]"},
            "outputs": {"motion": "float32 [1, 2L, 14] canonical units, channel order = channels"},
            "bytes": report["vq_decoder"]["bytes"],
        },
        "bigram": {"file": "bigram.bin", "dtype": "float16", "shape": [model.n_codes, model.n_codes],
                   "meaning": "log P(next | prev), row = prev"},
        "sampling": {"temperature": 0.8, "bigram_weight": 0.5,
                     "rule": "softmax(logits / temperature + bigram_weight * bigram[prev]) per step, prev = last sampled code"},
        "smoothing": {"kind": "zero-phase butterworth order 2", "cutoff_hz": DEFAULT_SMOOTH_HZ},
        "neutral": {"eye_open_l": 0.6, "eye_open_r": 0.6, "gaze_yaw": 0.0, "gaze_pitch": 0.0},
        "retrieval": {"file": "retrieval.json", "bin": "retrieval.bin"} if index is not None else None,
        "default_backend": verdict.get("default_backend", "retrieval"),
        "verdict": verdict,
        "training": model.info.get("training", {}),
    }
    if index is not None:
        b, j = index.save(out_dir, "retrieval")
        report["retrieval"] = {"bin": b, "json": j, "bytes": os.path.getsize(b) + os.path.getsize(j), "n_windows": len(index)}
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as fh:
        json.dump(model_json, fh, indent=1)
    report["model_json"] = os.path.join(out_dir, "model.json")
    report["total_bytes"] = sum(v.get("bytes", 0) for v in report.values() if isinstance(v, dict))
    return report
