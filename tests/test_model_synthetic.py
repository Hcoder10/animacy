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
            "--epochs-vq", "8", "--epochs-a2m", "6", "--export-dir", export_dir, "--device", "cpu", "--seed", "0"]
    t0 = time.time()
    rc = train.main(argv)
    assert rc == 0
    return {"out": out, "export_dir": export_dir, "seconds": time.time() - t0}


def test_train_writes_checkpoints_and_metrics(trained):
    out = trained["out"]
    for f in ("vq.pt", "a2m.pt", "model_info.json", "metrics.json", "REPORT.md", "retrieval.json", "retrieval.bin"):
        assert os.path.exists(os.path.join(out, f)), f
    m = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8"))
    assert m["data"]["n_clips"] == 6
    assert m["split"]["mode"] == "subject" and not m["split"]["leaky"]
    assert m["vq"]["used_codes_train"] > 8, "codebook collapsed"
    ev = m["eval"]
    assert "codes" in ev and ev["codes"]["nll_model"] > 0
    for cond in ("model", "model_shuffled", "model_causal", "retrieval", "retrieval_shuffled"):
        assert cond in ev["beat"] and cond in ev["stillness"] and cond in ev["velocity"], cond
        assert cond in ev["beat_all_channels"], cond
    assert ev["verdict"]["default_backend"] in ("model", "retrieval")
    for k, v in ev["legality"].items():
        assert v["violations"] == 0, (k, v)
    assert trained["seconds"] < 180, f"synthetic pipeline took {trained['seconds']:.0f}s"


def test_export_matches_torch(trained):
    m = json.load(open(os.path.join(trained["out"], "metrics.json"), encoding="utf-8"))
    ex = m["export"]
    for k in ("a2m", "vq_decoder"):
        assert ex[k]["ok"], ex[k]
        assert ex[k]["max_abs_diff"] < 1e-4
        assert os.path.exists(ex[k]["path"])
    for f in ("a2m.onnx", "vq_decoder.onnx", "model.json", "bigram.bin", "retrieval.json", "retrieval.bin"):
        assert os.path.exists(os.path.join(trained["export_dir"], f)), f
    from animacy.model.data import MODEL_CHANNELS

    mj = json.load(open(os.path.join(trained["export_dir"], "model.json"), encoding="utf-8"))
    assert mj["channels"] == MODEL_CHANNELS
    assert len(mj["stats"]["mean"]) == 14 and mj["n_codes"] == 512
    assert os.path.getsize(os.path.join(trained["export_dir"], "bigram.bin")) == 512 * 512 * 2
    assert ex["a2m"]["bytes"] + ex["vq_decoder"]["bytes"] < 10 * 1024 * 1024


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
    # retrieval baseline on the same input
    idx = RetrievalIndex.load(os.path.join(trained["out"], "retrieval.json"))
    m = idx.query(feats, speaking)
    assert m.shape == (T, 14) and np.isfinite(m).all()
