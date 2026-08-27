"""End-to-end: synthetic clips -> VQ -> a2m -> metrics -> ONNX, with tiny epochs.

    C:/Users/sarta/reachy-duplex/.venv/Scripts/python.exe -m pytest tests/test_model_synthetic.py -q

Must finish in < 3 minutes on CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from animacy.model import train

    out = str(tmp_path_factory.mktemp("ckpt"))
    export_dir = os.path.join(out, "web_models")
    argv = ["--synthetic", "--synthetic-clips", "6", "--synthetic-seconds", "12", "--out", out,
            "--epochs-vq", "8", "--epochs-a2m", "6", "--epochs-ar", "6", "--arch", "both", "--no-fp16",
            "--export-dir", export_dir, "--device", "cpu", "--seed", "0", "--cache-dir", ""]
    t0 = time.time()
    rc = train.main(argv)
    assert rc == 0
    return {"out": out, "export_dir": export_dir, "seconds": time.time() - t0}


def test_train_writes_checkpoints_and_metrics(trained):
    out = trained["out"]
    for f in ("vq.pt", "a2m.pt", "a2m_ar.pt", "model_info.json", "metrics.json", "REPORT.md", "retrieval.json", "retrieval.bin"):
        assert os.path.exists(os.path.join(out, f)), f
    m = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8"))
    assert m["data"]["n_clips"] == 6
    assert m["split"]["mode"] == "subject" and not m["split"]["leaky"]
    assert m["vq"]["used_codes_train"] > 8, "codebook collapsed"
    assert m["ar"]["n_train_chunks"] > 0 and m["a2m"]["n_train_chunks"] > 0
    ev = m["eval"]
    assert "codes" in ev and ev["codes"]["nll_model"] > 0 and ev["codes"]["nll_ar"] > 0
    for cond in ("model", "model_shuffled", "model_causal", "ar", "ar_shuffled", "ar_causal", "retrieval", "retrieval_shuffled"):
        assert cond in ev["beat"] and cond in ev["stillness"] and cond in ev["velocity"], cond
        assert cond in ev["beat_all_channels"], cond
    assert set(ev["verdict"]["candidates"]) == {"ff", "ar"}
    assert ev["verdict"]["default_backend"] in ("ar", "model", "retrieval")
    for k, v in ev["legality"].items():
        assert v["violations"] == 0, (k, v)
    assert trained["seconds"] < 180, f"synthetic pipeline took {trained['seconds']:.0f}s"


def test_export_matches_torch(trained):
    m = json.load(open(os.path.join(trained["out"], "metrics.json"), encoding="utf-8"))
    ex = m["export"]
    for k in ("a2m", "a2m_ar", "vq_decoder"):
        assert ex[k]["ok"], ex[k]
        assert ex[k]["max_abs_diff"] < 1e-4
        assert os.path.exists(ex[k]["path"])
    for f in ("a2m.onnx", "a2m_ar.onnx", "vq_decoder.onnx", "model.json", "bigram.bin", "retrieval.json", "retrieval.bin"):
        assert os.path.exists(os.path.join(trained["export_dir"], f)), f
    from animacy.model.data import MODEL_CHANNELS

    mj = json.load(open(os.path.join(trained["export_dir"], "model.json"), encoding="utf-8"))
    assert mj["channels"] == MODEL_CHANNELS
    assert len(mj["stats"]["mean"]) == 14 and mj["n_codes"] == 512
    assert mj["a2m_ar"]["arch"] == "ar" and mj["a2m_ar"]["bos"] == 512 and mj["default_arch"] in ("ff", "ar")
    assert os.path.getsize(os.path.join(trained["export_dir"], "bigram.bin")) == 512 * 512 * 2
    assert ex["a2m"]["bytes"] + ex["vq_decoder"]["bytes"] < 10 * 1024 * 1024


def test_fp16_export_keeps_the_argmax(trained, tmp_path):
    """float16 weights halve the file; the fp32-graph export stays exact and the most likely
    code is unchanged at nearly every position (the diff itself is reported, not asserted tight)."""
    from animacy.model.export import export_a2m, export_ar
    from animacy.model.infer import MotionModel

    model = MotionModel.load(trained["out"], "cpu")
    r_ar = export_ar(model, str(tmp_path / "a2m_ar.onnx"), fp16=True)
    r_ff = export_a2m(model, str(tmp_path / "a2m.onnx"), fp16=True)
    for r in (r_ar, r_ff):
        assert r["fp16"] and r["max_abs_diff_fp32_graph"] < 1e-4
        assert r["bytes"] < 0.6 * r["bytes_fp32"], (r["bytes"], r["bytes_fp32"])
        assert r["argmax_agreement"] >= 0.95, r
        assert r["max_abs_diff"] < 0.1, r["max_abs_diff"]


def test_stay_bias_makes_quiet_stretches_stiller(trained):
    from animacy.model.infer import MotionModel

    model = MotionModel.load(trained["out"], "cpu")
    rng = np.random.default_rng(3)
    L = 60
    f = rng.normal(size=(L, 66)).astype(np.float32)
    f[:, 64] = -1.0                                 # everything "quiet"
    s = np.zeros(L, np.int64)
    a = model.ar.generate(f, s, temperature=1.0, top_p=1.0, seed=1, stay_bias=0.0)
    b = model.ar.generate(f, s, temperature=1.0, top_p=1.0, seed=1, stay_bias=6.0)
    assert (b[1:] == b[:-1]).mean() >= (a[1:] == a[:-1]).mean()


def test_intent_rule_and_retrieval_bonus(trained):
    from animacy.features import audio_features
    from animacy.model.data import make_synthetic_clip
    from animacy.model.infer import MotionModel, generate, retrieve
    from animacy.model.intent import EXAMPLE_LINES, LEXICON, analyse
    from animacy.model.retrieval import RetrievalIndex

    n_total = sum(len(v) for v in EXAMPLE_LINES.values())
    n_ok = sum(analyse(s).tag == tag for tag, lines in EXAMPLE_LINES.items() for s in lines)
    assert n_total >= 30 and n_ok >= 27, f"{n_ok}/{n_total} example lines tagged as intended: " + str(
        [(s, analyse(s).tag) for tag, lines in EXAMPLE_LINES.items() for s in lines if analyse(s).tag != tag])
    try:
        from animacy.grade.movements import MOVEMENTS
    except Exception:  # noqa: BLE001
        MOVEMENTS = []
    for mv in MOVEMENTS:                                   # the grader's lines, never stored here
        assert analyse(mv.text).tag == mv.key, (mv.key, mv.text, analyse(mv.text).hits)
    ex, th = analyse(EXAMPLE_LINES["excitement"][0]), analyse(EXAMPLE_LINES["thinking"][0])
    assert ex.arousal > th.arousal and ex.amplitude > 1.0 > th.amplitude
    assert analyse("Yes.", override="excitement").tag == "excitement"
    # no multi-word lexicon phrase may be a verbatim 4+ word span of a grader line
    for mv in MOVEMENTS:
        low = mv.text.lower()
        for phrases in LEXICON.values():
            for p in phrases:
                assert not (len(p.split()) >= 4 and p in low), (p, mv.text)
    idx = RetrievalIndex.load(os.path.join(trained["out"], "retrieval.json"))
    assert idx.arousal is not None and len(idx.arousal) == len(idx) and 0 <= idx.arousal.min() <= idx.arousal.max() <= 1
    assert idx.proto is not None and set(idx.proto) == {"agreement", "doubt", "excitement", "thinking", "greeting"}
    assert all(len(v) == len(idx) and 0 <= v.min() <= v.max() <= 1 for v in idx.proto.values())
    model = MotionModel.load(trained["out"], "cpu")
    frames, wav = make_synthetic_clip(seed=77, seconds=4.0, subject="synthX")
    T = len(frames)
    feats = audio_features(wav, 16000, n_ticks=T)
    speaking = frames["speaking"].to_numpy()
    calm = retrieve(idx, feats, speaking, model, intent="thinking")
    loud = retrieve(idx, feats, speaking, model, intent="excitement", proto_weight=0.25, energy_floor=0.5)
    assert calm.validate() == [] and loud.validate() == []
    assert calm.meta["intent"]["tag"] == "thinking" and loud.meta["amplitude"] > calm.meta["amplitude"]
    assert loud.meta["proto_mean"] is not None and loud.meta["energy"] is not None
    off = retrieve(idx, feats, speaking, model, intent="excitement", proto_weight=0.0, energy_floor=0)
    assert off.meta["proto_weight"] == 0.0 and off.validate() == []
    clip = generate(model, feats, speaking, seed=1, intent="No way, that is incredible news!")
    assert clip.validate() == [] and clip.meta["intent"]["tag"] == "excitement" and clip.meta["amplitude"] > 1.0
    # the audio-only proxy runs end to end too
    assert idx.query(feats, speaking, use_audio_arousal=True).shape == (T, 14)


def test_server_index_uncapped_and_fp16_in_ram(trained, tmp_path):
    """max_windows=0 keeps every window (the server-side index); float16 motion in RAM
    gives the same query result as float32 up to half precision."""
    from animacy.model.data import load_clips
    from animacy.model.retrieval import RetrievalIndex

    clips = load_clips(os.path.join(trained["out"], "synthetic_clips"), verbose=False)
    full = RetrievalIndex.build(clips, max_windows=0)
    capped = RetrievalIndex.build(clips, max_windows=50)
    assert len(full) == full.meta["n_source_windows"] > 50 == len(capped)
    assert full.memory_estimate(len(full))["total_mb"] > 0
    full.save(str(tmp_path), "retrieval")
    a = RetrievalIndex.load(str(tmp_path / "retrieval.json"))
    b = RetrievalIndex.load(str(tmp_path / "retrieval.json"), motion_fp16=True)
    assert b.motion.dtype == np.float16 and len(a) == len(b) == len(full)
    c = clips[0]
    ma, mb = a.query(c.features[:90], c.speaking[:90]), b.query(c.features[:90], c.speaking[:90])
    assert ma.shape == mb.shape == (90, 14) and np.abs(ma - mb).max() < 0.05


def test_ar_onnx_step_matches_python_generation(trained):
    """The browser loop (ONNX single-step, full recompute) reproduces AudioToMotionAR.generate
    exactly when fed the same samples, in talk and listen mode."""
    import onnxruntime as ort

    from animacy.model.infer import MotionModel

    model = MotionModel.load(trained["out"], "cpu")
    ar = model.ar
    sess = ort.InferenceSession(os.path.join(trained["export_dir"], "a2m_ar.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(5)
    L = 45
    f = rng.normal(size=(L, 66)).astype(np.float32)
    s = rng.integers(0, 2, size=L).astype(np.int64)
    for causal in (False, True):
        codes_py = ar.generate(f, s, causal=causal, temperature=0.8, top_p=0.9, seed=11)
        # replay: with the same rng stream the ONNX logits must lead to the same choices
        rng2 = np.random.default_rng(11)
        hist = [ar.bos]
        for t in range(L):
            nxt, _ = sess.run(None, {"features": f[None], "speaking": s[None], "causal": np.array([int(causal)], np.int64),
                                     "codes": np.asarray(hist, np.int64)[None]})
            c = ar.sample(nxt[0], 0.8, 0.9, rng2)
            hist.append(c)
        assert np.array_equal(np.asarray(hist[1:]), codes_py), "ONNX step loop diverged from the Python generator"
    # listen mode is causal in the audio: changing future audio must not change earlier logits
    f2 = f.copy()
    f2[30:] += 5.0
    hist = np.concatenate([[ar.bos], codes_py[:19]]).astype(np.int64)[None]
    a = sess.run(None, {"features": f[None], "speaking": s[None], "causal": np.array([1], np.int64), "codes": hist})[1]
    b = sess.run(None, {"features": f2[None], "speaking": s[None], "causal": np.array([1], np.int64), "codes": hist})[1]
    assert np.abs(a - b).max() < 1e-4


def test_onnx_dynamic_length_and_decoder_roundtrip(trained):
    import onnxruntime as ort

    from animacy.model.infer import MotionModel

    model = MotionModel.load(trained["out"], "cpu")
    sess = ort.InferenceSession(os.path.join(trained["export_dir"], "a2m.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    for L in (7, 150):
        f = rng.normal(size=(1, L, 66)).astype(np.float32)
        s = rng.integers(0, 2, size=(1, L)).astype(np.int64)
        for c in (0, 1):
            ref = model.a2m.logits(f[0], s[0], causal=bool(c))
            got = sess.run(None, {"features": f, "speaking": s, "causal": np.array([c], np.int64)})[0][0]
            assert got.shape == (L, 512)
            assert np.abs(got - ref).max() < 1e-4
    # causal really is causal: the past does not depend on the future
    L = 40
    f = rng.normal(size=(1, L, 66)).astype(np.float32)
    s = np.zeros((1, L), np.int64)
    a = sess.run(None, {"features": f, "speaking": s, "causal": np.array([1], np.int64)})[0][0]
    f2 = f.copy()
    f2[0, 20:] += 5.0
    b = sess.run(None, {"features": f2, "speaking": s, "causal": np.array([1], np.int64)})[0][0]
    assert np.abs(a[:20] - b[:20]).max() < 1e-4
    assert np.abs(a[20:] - b[20:]).max() > 1e-3
    dec = ort.InferenceSession(os.path.join(trained["export_dir"], "vq_decoder.onnx"), providers=["CPUExecutionProvider"])
    codes = rng.integers(0, 512, size=(1, 23)).astype(np.int64)
    motion = dec.run(None, {"codes": codes})[0]
    assert motion.shape == (1, 46, 14)
    py = model.vq.denormalise(model.vq.decode(codes[0]))
    assert np.abs(motion[0] - py).max() < 1e-4


def test_generate_returns_valid_clip(trained):
    from animacy.features import audio_features
    from animacy.model.data import make_synthetic_clip
    from animacy.model.infer import MotionModel, generate
    from animacy.model.retrieval import RetrievalIndex

    model = MotionModel.load(trained["out"], "cpu")
    frames, wav = make_synthetic_clip(seed=999, seconds=5.5, subject="synthX")
    T = len(frames)
    feats = audio_features(wav, 16000, n_ticks=T)
    speaking = frames["speaking"].to_numpy()
    clip = generate(model, feats, speaking, causal=False, seed=3)
    assert clip.validate() == []
    assert len(clip) == T
    assert float(clip.frames["face_valid"].min()) == 1.0
    assert np.array_equal(clip.frames["speaking"].to_numpy(), (speaking > 0).astype(np.float32))
    # deterministic given the seed, different for another seed
    again = generate(model, feats, speaking, causal=False, seed=3)
    assert np.array_equal(clip.frames["head_pitch"].to_numpy(), again.frames["head_pitch"].to_numpy())
    other = generate(model, feats, speaking, causal=True, seed=4)
    assert other.validate() == []
    ar_clip = generate(model, feats, speaking, causal=False, seed=4, arch="ar")
    assert ar_clip.validate() == [] and len(ar_clip) == T and ar_clip.meta["arch"] == "ar"
    ar_again = generate(model, feats, speaking, causal=False, seed=4, arch="ar")
    assert np.array_equal(ar_clip.frames["head_pitch"].to_numpy(), ar_again.frames["head_pitch"].to_numpy())
    # retrieval baseline on the same input
    idx = RetrievalIndex.load(os.path.join(trained["out"], "retrieval.json"))
    m = idx.query(feats, speaking)
    assert m.shape == (T, 14) and np.isfinite(m).all()
