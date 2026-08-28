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
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..features import N_FEATS
from .data import DETREND_HZ, FEATURE_CONTRACT, FRAMES_PER_CODE, MODEL_CHANNELS, NORM_CLIP, POSE_CHANNELS
from .infer import DEFAULT_SMOOTH_HZ, ENERGY_CHANNELS, MotionModel
from .retrieval import RetrievalIndex

OPSET = 17


class _A2MWrapper(nn.Module):
    def __init__(self, a2m: nn.Module) -> None:
        super().__init__()
        self.a2m = a2m

    def forward(self, features: torch.Tensor, speaking: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        return self.a2m(features, speaking, causal)


class _ARWrapper(nn.Module):
    """One AR step, full recompute: audio + BOS-prefixed code history -> logits for the next code
    (and the teacher-forced logits for every history position)."""

    def __init__(self, ar: nn.Module) -> None:
        super().__init__()
        self.ar = ar

    def forward(self, features, speaking, causal, codes):
        logits_all = self.ar(features, speaking, causal, codes)          # [1, t, n_codes]
        return logits_all[:, -1, :], logits_all


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


def to_fp16_initializers(path: str, min_elems: int = 64) -> Dict:
    """Store every float32 initializer with >= ``min_elems`` values as float16 and Cast it
    back at load: compute stays float32, the file halves. Returns size before/after."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    before = os.path.getsize(path)
    m = onnx.load(path)
    g = m.graph
    inits, casts, n_conv = [], [], 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            arr = numpy_helper.to_array(init).astype(np.float16)
            name16 = init.name + "__fp16"
            inits.append(numpy_helper.from_array(arr, name16))
            casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name="cast__" + init.name))
            n_conv += 1
        else:
            inits.append(init)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.save(m, path)
    return {"bytes_before": before, "bytes_after": os.path.getsize(path), "n_converted": n_conv}


def _ort_session(path: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _verify_a2m(wrapper, path, verify_lengths, rng):
    """(max abs logits diff, position-wise argmax agreement) of the graph vs torch."""
    sess = _ort_session(path)
    worst, agree, n = 0.0, 0, 0
    for L in verify_lengths:
        f = rng.normal(size=(1, L, N_FEATS)).astype(np.float32)
        s = rng.integers(0, 2, size=(1, L)).astype(np.int64)
        for c in (0, 1):
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(f), torch.from_numpy(s), torch.tensor([c], dtype=torch.int64)).numpy()
            got = sess.run(None, {"features": f, "speaking": s, "causal": np.array([c], np.int64)})[0]
            assert got.shape == ref.shape, (got.shape, ref.shape)
            worst = max(worst, float(np.abs(got - ref).max()))
            agree += int((got.argmax(-1) == ref.argmax(-1)).sum())
            n += int(np.prod(ref.shape[:-1]))
    return worst, agree / max(n, 1)


TOL_FP16 = 5e-3     # measured on the real v1ar: 1.1e-3 (a2m) / 2.3e-3 (a2m_ar) on logits of magnitude ~10,
                    # with 100 % identical sampled codes vs torch; 1e-3 was too tight for a 4-layer fp16-weight stack


def export_a2m(model: MotionModel, path: str, verify_lengths=(30, 97), tol: float = 1e-4, fp16: bool = False,
               tol_fp16: float = TOL_FP16) -> Dict:
    a2m = model.a2m.eval().cpu()
    wrapper = _A2MWrapper(a2m).eval()
    L0 = 30
    feats = torch.randn(1, L0, N_FEATS)
    spk = torch.randint(0, 2, (1, L0), dtype=torch.int64)
    causal = torch.tensor([0], dtype=torch.int64)
    exporter = _export(wrapper, (feats, spk, causal), path, ["features", "speaking", "causal"], ["logits"],
                       {"features": {1: "L"}, "speaking": {1: "L"}, "logits": {1: "L"}})
    worst, agree = _verify_a2m(wrapper, path, verify_lengths, np.random.default_rng(0))
    rep = {"path": path, "bytes": os.path.getsize(path), "exporter": exporter, "max_abs_diff": worst, "ok": worst < tol,
           "argmax_agreement": agree, "verify_lengths": list(verify_lengths), "tol": tol, "fp16": False}
    if fp16:
        conv = to_fp16_initializers(path)
        worst16, agree16 = _verify_a2m(wrapper, path, verify_lengths, np.random.default_rng(0))
        rep.update({"fp16": True, "bytes": conv["bytes_after"], "bytes_fp32": conv["bytes_before"],
                    "max_abs_diff_fp32_graph": worst, "max_abs_diff": worst16, "argmax_agreement": agree16,
                    "tol": tol_fp16, "ok": worst16 < tol_fp16,
                    "fp16_note": "logits diff of the fp16-initializer graph vs torch; the fp32 graph diff is max_abs_diff_fp32_graph; "
                                 "argmax_agreement = fraction of positions whose most likely code is unchanged"})
    return rep


def _ar_code_agreement(ar, sess, L: int = 45, seed: int = 11) -> Dict:
    """Run the ONNX step loop and the Python generator with the same RNG stream: fraction of
    identical codes (talk + listen). The decision-relevant check for a reduced-precision graph."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(L, N_FEATS)).astype(np.float32)
    s = rng.integers(0, 2, size=L).astype(np.int64)
    out = {}
    for causal in (0, 1):
        codes_py = ar.generate(f, s, causal=bool(causal), temperature=0.8, top_p=0.9, seed=seed)
        rng2 = np.random.default_rng(seed)
        hist = [ar.bos]
        for _ in range(L):
            nxt, _ = sess.run(None, {"features": f[None], "speaking": s[None], "causal": np.array([causal], np.int64),
                                     "codes": np.asarray(hist, np.int64)[None]})
            hist.append(ar.sample(nxt[0], 0.8, 0.9, rng2, prev=hist[-1] if len(hist) > 1 else None))
        out["listen" if causal else "talk"] = float(np.mean(np.asarray(hist[1:]) == codes_py))
    return out


def export_ar(model: MotionModel, path: str, verify_cases=((30, 1), (97, 40), (60, 60)), tol: float = 1e-4,
              fp16: bool = False, tol_fp16: float = TOL_FP16) -> Dict:
    """``a2m_ar.onnx``: features [1, L, 66], speaking [1, L], causal [1], codes [1, t] (BOS then the
    codes so far) -> logits_next [1, n_codes], logits_all [1, t, n_codes]. Verified against torch at
    several (L, t) and against the windowed Python generator path."""
    ar = model.ar.eval().cpu()
    wrapper = _ARWrapper(ar).eval()
    L0, t0 = 30, 5
    feats = torch.randn(1, L0, N_FEATS)
    spk = torch.randint(0, 2, (1, L0), dtype=torch.int64)
    causal = torch.tensor([0], dtype=torch.int64)
    codes = torch.cat([torch.tensor([[ar.bos]]), torch.randint(0, ar.n_codes, (1, t0 - 1))], dim=1)
    exporter = _export(wrapper, (feats, spk, causal, codes), path, ["features", "speaking", "causal", "codes"],
                       ["logits_next", "logits_all"],
                       {"features": {1: "L"}, "speaking": {1: "L"}, "codes": {1: "t"}, "logits_all": {1: "t"}})
    rep = _verify_ar(ar, wrapper, path, verify_cases)
    rep.update({"path": path, "bytes": os.path.getsize(path), "exporter": exporter, "tol": tol, "ok": rep["max_abs_diff"] < tol,
                "verify_lengths": [list(x) for x in verify_cases], "fp16": False})
    if fp16:
        conv = to_fp16_initializers(path)
        rep16 = _verify_ar(ar, wrapper, path, verify_cases)
        rep.update({"fp16": True, "bytes": conv["bytes_after"], "bytes_fp32": conv["bytes_before"],
                    "max_abs_diff_fp32_graph": rep["max_abs_diff"], "max_abs_diff": rep16["max_abs_diff"],
                    "argmax_agreement": rep16["argmax_agreement"], "code_agreement_fp16": rep16["code_agreement"],
                    "tol": tol_fp16, "ok": rep16["max_abs_diff"] < tol_fp16,
                    "fp16_note": "logits diff of the fp16-initializer graph vs torch; argmax_agreement = fraction of positions "
                                 "whose most likely code is unchanged; code_agreement_fp16 = identical sampled codes vs the "
                                 "Python generator under the same RNG (a single early divergence changes the rest of the sequence)"})
    return rep


def _verify_ar(ar, wrapper, path, verify_cases) -> Dict:
    sess = _ort_session(path)
    worst, agree, n = 0.0, 0, 0
    rng = np.random.default_rng(0)
    for L, t in verify_cases:
        f = rng.normal(size=(1, L, N_FEATS)).astype(np.float32)
        s = rng.integers(0, 2, size=(1, L)).astype(np.int64)
        c = np.concatenate([[ar.bos], rng.integers(0, ar.n_codes, size=t - 1)]).astype(np.int64)[None]
        for cf in (0, 1):
            with torch.no_grad():
                ref_next, ref_all = wrapper(torch.from_numpy(f), torch.from_numpy(s), torch.tensor([cf], dtype=torch.int64), torch.from_numpy(c))
                # the windowed decode used by AudioToMotionAR.generate must equal the full recompute
                mem = ar.encode_audio(torch.from_numpy(f), torch.from_numpy(s), torch.tensor([cf]))
                ctx = ar.window * ar.dec_layers
                tail = c[0][-ctx:]
                win = ar.decode(mem, torch.from_numpy(tail)[None], torch.tensor([cf]), offset=len(c[0]) - len(tail))[0, -1]
            got_next, got_all = sess.run(None, {"features": f, "speaking": s, "causal": np.array([cf], np.int64), "codes": c})
            assert got_next.shape == (1, ar.n_codes) and got_all.shape == (1, t, ar.n_codes), (got_next.shape, got_all.shape)
            worst = max(worst, float(np.abs(got_next - ref_next.numpy()).max()), float(np.abs(got_all - ref_all.numpy()).max()),
                        float(np.abs(win.numpy() - ref_next.numpy()[0]).max()))
            agree += int((got_all.argmax(-1) == ref_all.numpy().argmax(-1)).sum())
            n += t
    return {"max_abs_diff": worst, "argmax_agreement": agree / max(n, 1), "code_agreement": _ar_code_agreement(ar, sess)}


def export_vq_decoder(model: MotionModel, path: str, verify_lengths=(15, 61), tol: float = 1e-4) -> Dict:
    vq = model.vq.eval().cpu()
    wrapper = _DecoderWrapper(vq).eval()
    codes = torch.randint(0, vq.n_codes, (1, 15), dtype=torch.int64)
    exporter = _export(wrapper, (codes,), path, ["codes"], ["motion"], {"codes": {1: "L"}, "motion": {1: "T"}})
    sess = _ort_session(path)
    worst_raw, worst_std = 0.0, 0.0
    std = vq.stats["std"][None, None, :]
    rng = np.random.default_rng(0)
    for L in verify_lengths:
        c = rng.integers(0, vq.n_codes, size=(1, L)).astype(np.int64)
        with torch.no_grad():
            ref = wrapper(torch.from_numpy(c)).numpy()
        got = sess.run(None, {"codes": c})[0]
        assert got.shape == (1, FRAMES_PER_CODE * L, len(MODEL_CHANNELS)), got.shape
        # the graph must also agree with the python decode path (denormalised)
        py = vq.denormalise(vq.decode(c[0]))[None]
        for other in (ref, py):
            d = np.abs(got - other)
            worst_raw = max(worst_raw, float(d.max()))
            worst_std = max(worst_std, float((d / std).max()))
    # outputs are in canonical units (mm, deg): the tolerance applies in standardised units,
    # the raw-unit difference (float32 accumulation order) is reported alongside
    ok = worst_std < tol
    return {"path": path, "bytes": os.path.getsize(path), "exporter": exporter, "max_abs_diff": worst_std,
            "max_abs_diff_units": "standardised (per-channel sd)", "max_abs_diff_raw_units": worst_raw, "ok": ok,
            "verify_lengths": list(verify_lengths), "tol": tol}


def _intent_block() -> Dict:
    from .intent import describe
    from .retrieval import AROUSAL_BONUS, PROTO_DOC, PROTO_WEIGHT, THINKING_BONUS

    d = describe(AROUSAL_BONUS, THINKING_BONUS)
    d["gesture_prototypes"] = {"weight": PROTO_WEIGHT, "fields": "retrieval.json: proto[tag][window] in 0..1", **PROTO_DOC}
    return d


def _amplitude_tiers() -> Dict:
    from .intent import AMPLITUDE_TIERS

    return dict(AMPLITUDE_TIERS)


def _placement_block(model: MotionModel) -> Dict:
    from .gesture import PlacementConfig

    cfg = PlacementConfig.from_any((model.info.get("postprocess") or {}).get("gesture_placement"))
    return {
        **cfg.to_dict(),
        "accents": "smoothed (9-frame Hann) feature[64] over the voiced span (speaking=1, else feature[64] > -0.3); peaks >= min_gap_s "
                   "apart with prominence >= 0.15; N = 1 (< 2.5 s of speech), 2 (2.5-5 s), 3 (> 5 s); the first peak after onset and "
                   "the last peak are always kept, the rest by height",
        "library": "retrieval.json: gestures[intent] = top-k windows by proto score with peak_t (frame of max head speed) and dur",
        "rule": "out = base * gain + sum(gestures): gain = base_gain_idle, base_gain_active under a gesture, hold_gain during a hold; "
                "each gesture = (segment - segment[0]) * tier * amplitude_boost, added with raised-cosine edges of blend_ms so its peak_t "
                "lands on the accent; a gesture is picked by a seeded draw among library entries whose dur fits the gap to the next "
                "accent (else the shortest third); thinking = one tilt-and-hold from the onset held until the last accent then released "
                "over 2 * blend; excitement = the rise on the first accent, held until the last accent, then dropped; then energy floor "
                "-> pitch floor -> settle -> clamp. enabled=false or 0 disables (A/B knob).",
    }


def export_bundle(model: MotionModel, index: Optional[RetrievalIndex], out_dir: str, metrics: Optional[Dict] = None,
                  tol: float = 1e-4, fp16: bool = False, archs: Optional[List[str]] = None, json_only: bool = False) -> Dict:
    """Write ``a2m.onnx``, ``a2m_ar.onnx``, ``vq_decoder.onnx``, ``bigram.bin``, ``model.json`` (+ retrieval
    index). ``fp16`` stores the predictors' weights as float16 (compute stays float32). ``archs``
    restricts which predictors ship (e.g. ``["ar"]`` drops ``a2m.onnx``; a stale file is removed).
    ``json_only`` rewrites ``model.json`` and leaves every other file in ``out_dir`` untouched."""
    os.makedirs(out_dir, exist_ok=True)
    archs = [a for a in (archs or model.archs) if a in model.archs]
    report: Dict = {"archs": archs}
    a2m_path = os.path.join(out_dir, "a2m.onnx")
    ar_path = os.path.join(out_dir, "a2m_ar.onnx")
    vq_path = os.path.join(out_dir, "vq_decoder.onnx")
    if json_only:
        # rewrite model.json only; every binary in out_dir stays byte-identical
        def _existing(path):
            return {"path": path, "bytes": os.path.getsize(path), "fp16": fp16, "exporter": "unchanged", "max_abs_diff": None, "ok": True}

        if "ff" in archs and os.path.exists(a2m_path):
            report["a2m"] = _existing(a2m_path)
        if "ar" in archs and os.path.exists(ar_path):
            report["a2m_ar"] = _existing(ar_path)
        report["vq_decoder"] = _existing(vq_path)
    else:
        if model.a2m is not None and "ff" in archs:
            report["a2m"] = export_a2m(model, a2m_path, tol=tol, fp16=fp16)
        elif os.path.exists(a2m_path):
            os.remove(a2m_path)
        if model.ar is not None and "ar" in archs:
            report["a2m_ar"] = export_ar(model, ar_path, tol=tol, fp16=fp16)
        report["vq_decoder"] = export_vq_decoder(model, vq_path, tol=tol)

    bigram_path = os.path.join(out_dir, "bigram.bin")
    if model.bigram_logp is not None and not json_only:
        bigram16 = np.asarray(model.bigram_logp, np.float32).astype(np.float16)
        with open(bigram_path, "wb") as fh:
            fh.write(bigram16.tobytes())
        report["bigram"] = {"path": bigram_path, "bytes": os.path.getsize(bigram_path)}
    elif os.path.exists(bigram_path):
        report["bigram"] = {"path": bigram_path, "bytes": os.path.getsize(bigram_path)}
    else:
        report["bigram"] = {"path": None, "bytes": 0}

    stats = model.vq.stats
    verdict = (metrics or {}).get("verdict", {})
    default_arch = verdict.get("default_arch") or model.info.get("default_arch") or ("ar" if model.ar is not None else "ff")
    if default_arch not in archs and archs:
        default_arch = archs[0]
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
        "archs": archs,
        "default_arch": default_arch,
        "a2m": {
            "arch": "ff",
            "file": "a2m.onnx",
            "inputs": {"features": "float32 [1, L, 66] (30 Hz features averaged in pairs -> 15 Hz)",
                       "speaking": "int64 [1, L] (any of the pair)",
                       "causal": "int64 [1]: 0 = talk (non-causal), 1 = listen (causal)"},
            "outputs": {"logits": "float32 [1, L, 512]"},
            "weights": "float16 initializers, float32 compute" if report["a2m"].get("fp16") else "float32",
            "bytes": report["a2m"]["bytes"],
        } if "a2m" in report else None,
        "a2m_ar": {
            "arch": "ar",
            "file": "a2m_ar.onnx",
            "bos": model.n_codes,
            "window": model.ar.window,
            "dec_layers": model.ar.dec_layers,
            "inputs": {"features": "float32 [1, L, 66] (15 Hz, same as a2m)",
                       "speaking": "int64 [1, L]",
                       "causal": "int64 [1]: 0 = talk, 1 = listen (causal audio trunk AND causal cross-attention)",
                       "codes": "int64 [1, t]: BOS (= n_codes) followed by the codes sampled so far; t >= 1, t <= L"},
            "outputs": {"logits_next": "float32 [1, 512]: logits for the code at position t-1 (the next code to sample)",
                        "logits_all": "float32 [1, t, 512]: teacher-forced logits for positions 0..t-1"},
            "step": "single-step graph with full recompute: codes = [BOS]; for i in 0..L-1: run(features, speaking, causal, codes) -> "
                    "logits_next[prev] -= repeat_penalty (prev = last sampled code, skip at i = 0); "
                    "if features[i][64] < stay_energy: logits_next[prev] += stay_bias (feature 64 = normalised log energy) -> "
                    "sample softmax(logits / temperature) with top-p -> append. Passing the whole history is exact "
                    "(the decoder's self-attention window is `window` codes per layer).",
            "sampling": {"temperature": model.info.get("sampling", {}).get("temperature", 0.8),
                         "top_p": model.info.get("sampling", {}).get("top_p", 0.9),
                         "repeat_penalty": model.info.get("sampling", {}).get("repeat_penalty", 0.0),
                         "stay_bias": model.info.get("sampling", {}).get("stay_bias", 0.0),
                         "stay_energy": model.info.get("sampling", {}).get("stay_energy", -0.3)},
            "weights": "float16 initializers, float32 compute" if report["a2m_ar"].get("fp16") else "float32",
            "bytes": report["a2m_ar"]["bytes"],
        } if "a2m_ar" in report else None,
        "vq_decoder": {
            "file": "vq_decoder.onnx",
            "inputs": {"codes": "int64 [1, L]"},
            "outputs": {"motion": "float32 [1, 2L, 14] canonical units, channel order = channels"},
            "bytes": report["vq_decoder"]["bytes"],
        },
        "bigram": {"file": "bigram.bin", "dtype": "float16", "shape": [model.n_codes, model.n_codes],
                   "meaning": "log P(next | prev), row = prev (used by the feed-forward a2m sampler only)"} if model.bigram_logp is not None else None,
        "sampling": {"temperature": 0.8, "bigram_weight": 0.5,
                     "rule": "ff: softmax(logits / temperature + bigram_weight * bigram[prev]) per step, prev = last sampled code; ar: see a2m_ar.step"},
        "smoothing": {"kind": "zero-phase butterworth order 2", "cutoff_hz": DEFAULT_SMOOTH_HZ},
        "postprocess": {
            **{"settle_s": 0.5, "pitch_floor": -3.0, "amplitude": 1.0, "proto_weight": 0.25, "energy_floor": 0.0,
               **(model.info.get("postprocess") or {})},
            "order": "retrieve (cosine + continuity + speaking + arousal bonus + proto_weight * proto[intent]) or decode+smooth -> "
                     "amplitude tier by intent -> energy floor (one scalar for the whole utterance, 1.0..2.0) -> pitch floor -> "
                     "settle (after speech ends) -> clamp to channel bounds",
            "amplitude_tiers": _amplitude_tiers(),
            "energy_floor_rule": "energy = RMS over frames and the 9 channels " + str(ENERGY_CHANNELS) + " of (motion / stats.std[channel]) "
                                 "after removing each channel's mean over the utterance; if energy < energy_floor, multiply all 14 channels by "
                                 "min(2.0, energy_floor / energy); energy_floor = the corpus 60th percentile of that RMS over 3 s windows "
                                 "(hop 1 s); 0 disables",
            "energy_channels": list(ENERGY_CHANNELS),
            "pitch_floor_rule": "low-pass head_pitch at 0.3 Hz (2nd-order butterworth, zero-phase); where that baseline is below pitch_floor add "
                                "(pitch_floor - baseline) - the baseline is lifted, the motion on top of it is untouched",
            "settle_rule": "end = last frame with speaking=1 (else last frame with feature[64] > -0.3, else clip end); only if end < T: "
                           "w ramps linearly 0->1 over settle_s seconds starting AT end and stays 1 after; motion *= (1 - w); never mid-utterance",
            "proto_rule": "proto_weight * retrieval.proto[intent][window] is added to every window's score; 0 disables (A/B knob)",
            "gesture_placement": _placement_block(model),
        },
        "detrend": {"channels": list(POSE_CHANNELS), "cutoff_hz": DETREND_HZ,
                    "meaning": "pose channels are the residual above cutoff_hz: output motion is centred on neutral; add the gaze overlay for where to look"},
        "neutral": {"eye_open_l": 0.6, "eye_open_r": 0.6, "gaze_yaw": 0.0, "gaze_pitch": 0.0},
        "retrieval": {"file": "retrieval.json", "bin": "retrieval.bin",
                      "intent_fields": index.arousal is not None,
                      "proto_fields": index.proto is not None,
                      "gesture_library": bool(getattr(index, "gestures", None))} if index is not None else None,
        "intent": _intent_block(),
        "default_backend": verdict.get("default_backend", "retrieval"),
        "verdict": verdict,
        "training": model.info.get("training", {}),
    }
    if index is not None and not json_only:
        b, j = index.save(out_dir, "retrieval")
        report["retrieval"] = {"bin": b, "json": j, "bytes": os.path.getsize(b) + os.path.getsize(j), "n_windows": len(index)}
    elif json_only and os.path.exists(os.path.join(out_dir, "retrieval.json")):
        b, j = os.path.join(out_dir, "retrieval.bin"), os.path.join(out_dir, "retrieval.json")
        report["retrieval"] = {"bin": b, "json": j, "bytes": os.path.getsize(b) + os.path.getsize(j), "n_windows": len(index) if index is not None else None}
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as fh:
        json.dump(model_json, fh, indent=1)
    report["model_json"] = os.path.join(out_dir, "model.json")
    report["total_bytes"] = sum(v.get("bytes", 0) for v in report.values() if isinstance(v, dict))
    return report


def main(argv=None) -> int:
    """``python -m animacy.model.export --ckpt checkpoints/v1 --out web/models``: re-export a
    trained checkpoint (its metrics.json, if present, supplies the verdict / default backend)."""
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--ckpt", default="checkpoints/v1")
    p.add_argument("--out", default="web/models")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true", help="float16 weights for a2m.onnx and a2m_ar.onnx")
    p.add_argument("--archs", nargs="*", default=None, help="which predictors to ship (default: all in the checkpoint), e.g. --archs ar")
    p.add_argument("--json-only", action="store_true", help="rewrite model.json only; ONNX / bigram / retrieval files stay untouched")
    a = p.parse_args(argv)
    model = MotionModel.load(a.ckpt, "cpu")
    idx_path = os.path.join(a.ckpt, "retrieval.json")
    index = RetrievalIndex.load(idx_path) if os.path.exists(idx_path) else None
    m_path = os.path.join(a.ckpt, "metrics.json")
    metrics = json.load(open(m_path, encoding="utf-8")) if os.path.exists(m_path) else None
    rep = export_bundle(model, index, a.out, (metrics or {}).get("eval"), tol=a.tol, fp16=a.fp16, archs=a.archs, json_only=a.json_only)
    ok = True
    for k in ("a2m", "a2m_ar", "vq_decoder"):
        if k not in rep:
            continue
        r = rep[k]
        ok &= bool(r["ok"])
        if a.json_only:
            print(f"{os.path.basename(r['path'])}: {r['bytes'] / 1e6:.2f} MB (unchanged)")
            continue
        extra = ""
        if r.get("fp16"):
            extra = f" [fp16 weights: {r['bytes_fp32'] / 1e6:.2f} -> {r['bytes'] / 1e6:.2f} MB, fp32-graph diff {r['max_abs_diff_fp32_graph']:.2e}" \
                    + (f", code agreement {r['code_agreement_fp16']}" if "code_agreement_fp16" in r else "") + "]"
        print(f"{os.path.basename(r['path'])}: {r['bytes'] / 1e6:.2f} MB ({r['exporter']}), max abs diff vs torch "
              f"{r['max_abs_diff']:.2e} (tol {r['tol']}) -> {'OK' if r['ok'] else 'MISMATCH'}{extra}")
    if "retrieval" in rep:
        print(f"retrieval: {rep['retrieval']['n_windows']} windows, {rep['retrieval']['bytes'] / 1e6:.2f} MB")
    print(f"bundle {rep['total_bytes'] / 1e6:.2f} MB -> {a.out}; default backend = "
          f"{json.load(open(rep['model_json'], encoding='utf-8'))['default_backend']}")
    if metrics is not None and not a.json_only:      # a json-only rewrite measured nothing new
        metrics["export"] = rep
        with open(m_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=1, default=float)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
