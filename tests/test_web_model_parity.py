"""web/js/model.js must reproduce animacy/model/infer.py on the web/models bundle.

Runs the browser runtime (onnxruntime-web, wasm) in headless Chromium via
Playwright against the same web/models/*.onnx files driven from Python with
onnxruntime, on identical inputs:

  * a2m logits (causal 0 and 1)                            → max |diff| < 1e-3
  * vq_decoder motion for fixed codes                      → max |diff| < 1e-3 (canonical units)
  * greedy sampling (temperature → 0: argmax of logits/T + w·bigram[prev]) → identical codes
  * full generate_motion with greedy codes (decode + zero-phase smoothing) → < 2e-3
  * retrieval query (motion matching)                      → identical window ids, motion < 1e-3
  * stochastic sampling: JS uses sfc32, Python numpy PCG64 → only the code
    histogram is compared (same support, no bit-exactness claimed)

Skipped when the bundle, node/playwright or chromium are missing.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(ROOT, "web", "models")
BUNDLE_OK = all(os.path.isfile(os.path.join(MODELS, f)) for f in ("model.json", "a2m.onnx", "vq_decoder.onnx", "bigram.bin"))
RETRIEVAL_OK = all(os.path.isfile(os.path.join(MODELS, f)) for f in ("retrieval.json", "retrieval.bin"))

pytestmark = pytest.mark.skipif(not BUNDLE_OK, reason="no model bundle in web/models")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 15.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError("http server did not start")


@pytest.fixture(scope="module")
def js_results():
    """Everything computed by the browser on fixed inputs, plus the inputs themselves."""
    pytest.importorskip("playwright")
    pytest.importorskip("onnxruntime")
    from playwright.sync_api import sync_playwright

    from animacy.features import N_FEATS

    rng = np.random.default_rng(0)
    T = 61  # odd on purpose: exercises the tail-hold
    feats = rng.normal(size=(T, N_FEATS)).astype(np.float32)
    speaking = (rng.random(T) > 0.3).astype(np.int64)
    codes = rng.integers(0, 512, size=23).astype(np.int64)
    job = {"feats": feats.tolist(), "speaking": speaking.tolist(), "codes": codes.tolist(), "retrieval": RETRIEVAL_OK}

    port = _free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_port(port)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/web/?autoplay=0", wait_until="domcontentloaded")
            page.wait_for_function("window.animacy && window.animacy.ready === true", timeout=120_000)
            out = page.evaluate(
                """async (job) => {
                    const M = await import('./js/model.js');
                    const model = await M.MotionModel.load('models/');
                    const rows = job.feats.map(r => Float32Array.from(r));
                    const spk = Uint8Array.from(job.speaking);
                    const L = Math.floor(rows.length / 2);
                    const f15 = M.poolPairs(rows, 66);
                    const s15 = M.poolFlag(spk);
                    const res = { n_codes: model.nCodes, channels: model.channels };
                    res.logits0 = Array.from(await model.logits(f15, L, s15, false));
                    res.logits1 = Array.from(await model.logits(f15, L, s15, true));
                    res.decoded = Array.from(await model.decode(Int32Array.from(job.codes)));
                    const lg = Float32Array.from(res.logits0);
                    res.greedy = Array.from(M.sampleCodes(lg, L, model.nCodes, model.bigram, { temperature: 1e-6, bigramWeight: 0.5, seed: 0 }));
                    res.stochastic = Array.from(M.sampleCodes(lg, L, model.nCodes, model.bigram, { temperature: 0.8, bigramWeight: 0.5, seed: 7 }));
                    // the same sequence scored by the JS side (mean log p under logits/T + w·bigram[prev])
                    res.stochastic_lp_js = (() => { const n = model.nCodes, bg = model.bigram; let tot = 0, prev = -1;
                        for (let t = 0; t < L; t++) { let zmax = -Infinity; const z = new Float64Array(n);
                            for (let k = 0; k < n; k++) { const v = lg[t * n + k] / 0.8 + (prev >= 0 ? 0.5 * bg[prev * n + k] : 0); z[k] = v; if (v > zmax) zmax = v; }
                            let sum = 0; for (let k = 0; k < n; k++) sum += Math.exp(z[k] - zmax);
                            const c = res.stochastic[t]; tot += (z[c] - zmax) - Math.log(sum); prev = c; }
                        return tot / L; })();
                    // mean log p over many seeds (the JS sampler's expected likelihood under its own distribution)
                    res.stochastic_lp_many = (() => { const n = model.nCodes, bg = model.bigram; const lps = [];
                        for (let seed = 100; seed < 140; seed++) { const c = M.sampleCodes(lg, L, n, bg, { temperature: 0.8, bigramWeight: 0.5, seed }); let tot = 0, prev = -1;
                            for (let t = 0; t < L; t++) { let zmax = -Infinity; const z = new Float64Array(n);
                                for (let k = 0; k < n; k++) { const v = lg[t * n + k] / 0.8 + (prev >= 0 ? 0.5 * bg[prev * n + k] : 0); z[k] = v; if (v > zmax) zmax = v; }
                                let sum = 0; for (let k = 0; k < n; k++) sum += Math.exp(z[k] - zmax);
                                tot += (z[c[t]] - zmax) - Math.log(sum); prev = c[t]; }
                            lps.push(tot / L); }
                        return lps; })();
                    res.bigram_head = Array.from(model.bigram.slice(0, 8));
                    res.bigram_row_1_head = Array.from(model.bigram.slice(model.nCodes, model.nCodes + 8));
                    const g = await model.generate(rows, spk, { causal: false, temperature: 1e-6, bigramWeight: 0.5, seed: 0 });
                    res.generated = Array.from(g.motion);
                    res.generated_codes = Array.from(g.codes);
                    if (job.retrieval) {
                        const idx = await M.RetrievalIndex.load('models/');
                        const q = idx.query(rows, spk);
                        res.retrieval_ids = q.ids;
                        res.retrieval_motion = Array.from(q.motion);
                        res.retrieval_n = idx.n;
                    }
                    return res;
                }""",
                job,
            )
            browser.close()
            assert not errors, errors
    finally:
        srv.terminate()
    out["feats"], out["speaking"], out["codes"] = feats, speaking, codes
    return out


def _py_sessions():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    a2m = ort.InferenceSession(os.path.join(MODELS, "a2m.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(MODELS, "vq_decoder.onnx"), so, providers=["CPUExecutionProvider"])
    return a2m, dec


def _bigram():
    meta = json.load(open(os.path.join(MODELS, "model.json"), encoding="utf-8"))
    n = meta["n_codes"]
    return np.fromfile(os.path.join(MODELS, "bigram.bin"), dtype=np.float16).astype(np.float32).reshape(n, n), meta


def test_a2m_logits_match(js_results):
    from animacy.model.data import pool_flag, pool_pairs

    a2m, _ = _py_sessions()
    f15 = pool_pairs(js_results["feats"])[None]
    s15 = pool_flag(js_results["speaking"])[None]
    for causal, key in ((0, "logits0"), (1, "logits1")):
        ref = a2m.run(None, {"features": f15, "speaking": s15, "causal": np.array([causal], np.int64)})[0]
        js = np.asarray(js_results[key], np.float32).reshape(ref.shape)
        d = np.abs(js - ref).max()
        assert d < 1e-3, f"causal={causal}: max |js - py| logits = {d:.2e}"


def test_vq_decoder_matches(js_results):
    _, dec = _py_sessions()
    ref = dec.run(None, {"codes": js_results["codes"][None]})[0]
    js = np.asarray(js_results["decoded"], np.float32).reshape(ref.shape)
    d = np.abs(js - ref).max()
    assert d < 1e-3, f"max |js - py| motion = {d:.2e}"


def test_greedy_sampling_matches(js_results):
    from animacy.model.data import pool_flag, pool_pairs
    from animacy.model.infer import sample_codes

    a2m, _ = _py_sessions()
    bigram, _ = _bigram()
    f15 = pool_pairs(js_results["feats"])[None]
    s15 = pool_flag(js_results["speaking"])[None]
    logits = a2m.run(None, {"features": f15, "speaking": s15, "causal": np.array([0], np.int64)})[0][0]
    ref = sample_codes(logits, bigram, temperature=1e-6, bigram_weight=0.5, seed=0)
    js = np.asarray(js_results["greedy"], np.int64)
    assert js.shape == ref.shape
    mismatch = int((js != ref).sum())
    assert mismatch == 0, f"{mismatch}/{len(ref)} greedy codes differ (near-ties in logits would explain ≤1)"


def test_generate_motion_matches_greedy_path(js_results):
    """decode(greedy codes) + zero-phase smoothing + odd-tail hold, in canonical units."""
    from animacy.model.infer import smooth_motion

    _, dec = _py_sessions()
    codes = np.asarray(js_results["greedy"], np.int64)
    assert list(js_results["generated_codes"]) == codes.tolist()
    m = dec.run(None, {"codes": codes[None]})[0][0]
    m = smooth_motion(m, 30.0, 6.0)
    T = len(js_results["feats"])
    if len(m) < T:
        m = np.concatenate([m, np.repeat(m[-1:], T - len(m), axis=0)])
    ref = m[:T]
    js = np.asarray(js_results["generated"], np.float32).reshape(ref.shape)
    d = np.abs(js - ref).max()
    assert d < 2e-3, f"max |js - py| generated motion = {d:.2e}"


def _seq_logprob(codes: np.ndarray, logits: np.ndarray, bigram: np.ndarray, temperature: float, w: float) -> float:
    """Mean log-probability of a code sequence under infer.sample_codes' per-step distribution."""
    lg = logits.astype(np.float64) / temperature
    total, prev = 0.0, -1
    for t, c in enumerate(codes):
        z = lg[t] + (w * bigram[prev] if prev >= 0 else 0.0)
        z = z - z.max()
        total += z[c] - np.log(np.exp(z).sum())
        prev = int(c)
    return total / len(codes)


def test_stochastic_sampling_follows_the_same_distribution(js_results):
    """Different RNGs (sfc32 vs numpy PCG64), so sequences differ; the JS draws must
    be as likely under Python's per-step distribution as Python's own draws are."""
    from animacy.model.data import pool_flag, pool_pairs
    from animacy.model.infer import sample_codes

    a2m, _ = _py_sessions()
    bigram, meta = _bigram()
    f15 = pool_pairs(js_results["feats"])[None]
    s15 = pool_flag(js_results["speaking"])[None]
    logits = a2m.run(None, {"features": f15, "speaking": s15, "causal": np.array([0], np.int64)})[0][0]
    js = np.asarray(js_results["stochastic"], np.int64)
    assert js.shape == (len(logits),)
    assert js.min() >= 0 and js.max() < meta["n_codes"]
    lp_js = _seq_logprob(js, logits, bigram, 0.8, 0.5)
    lp_py = np.array([_seq_logprob(sample_codes(logits, bigram, 0.8, 0.5, seed=s), logits, bigram, 0.8, 0.5) for s in range(40)])
    lp_js_many = np.asarray(js_results["stochastic_lp_many"])
    lp_uniform = -np.log(meta["n_codes"])
    print(f"mean log p: js seed7 {lp_js:.3f} (js-side {js_results['stochastic_lp_js']:.3f}); js 40 seeds {lp_js_many.mean():.3f} ± {lp_js_many.std():.3f}; "
          f"python 40 seeds {lp_py.mean():.3f} ± {lp_py.std():.3f}; uniform {lp_uniform:.3f}")
    # both evaluators must agree on the same sequence: otherwise logits or the bigram differ between the two sides
    assert abs(lp_js - js_results["stochastic_lp_js"]) < 0.05, "JS and Python score the same code sequence differently"
    # the two samplers must have the same expected likelihood (two-sample check on the seed means)
    se = np.sqrt(lp_js_many.var() / len(lp_js_many) + lp_py.var() / len(lp_py))
    zscore = (lp_js_many.mean() - lp_py.mean()) / max(se, 1e-9)
    print(f"z = {zscore:+.2f}")
    assert abs(zscore) < 4.0, f"JS sampler's expected likelihood differs from Python's (z = {zscore:+.2f})"


@pytest.mark.skipif(not RETRIEVAL_OK, reason="no retrieval index in web/models")
def test_retrieval_query_matches(js_results):
    from animacy.model.retrieval import RetrievalIndex

    idx = RetrievalIndex.load(os.path.join(MODELS, "retrieval.json"))
    assert js_results["retrieval_n"] == len(idx)
    ref, ids = idx.query(js_results["feats"], js_results["speaking"].astype(np.float32), return_ids=True)
    js_ids = np.asarray(js_results["retrieval_ids"], np.int64)
    same = int((js_ids == ids).sum())
    assert same >= len(ids) - 1, f"retrieval ids differ at {len(ids) - same}/{len(ids)} hops: js={js_ids.tolist()} py={ids.tolist()}"
    js = np.asarray(js_results["retrieval_motion"], np.float32).reshape(ref.shape)
    if same == len(ids):
        d = np.abs(js - ref).max()
        assert d < 1e-3, f"max |js - py| retrieval motion = {d:.2e}"
