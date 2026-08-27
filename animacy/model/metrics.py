"""Held-out evaluation, exactly the list in ``docs/MODEL.md``:

* code NLL / top-1 vs a unigram ("majority code") floor and vs retrieval;
* motion statistics: per-channel |velocity| histogram distance (Wasserstein-1)
  and stillness ratio vs ground truth, for model / retrieval / shuffled-audio;
* beat alignment: fraction of ground-truth head-velocity peaks within +-150 ms
  of a generated peak, and the same with block-shuffled audio (the model must
  beat its own shuffle);
* retarget legality: speed-cap violations after ``retarget_clip`` on every
  robot profile given.

Everything here is computed on whole contiguous runs of held-out clips.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..profile import Profile
from ..retarget import retarget_clip
from ..schema import RATE_HZ
from .data import FRAMES_PER_CODE, MODEL_CHANNELS, ClipData, normalise, pool_flag, pool_pairs
from .infer import MotionModel, generate_motion, motion_to_clip
from .retrieval import RetrievalIndex

HEAD_IDX = [MODEL_CHANNELS.index(c) for c in ("head_yaw", "head_pitch", "head_roll")]
STILL_DEG_PER_S = 5.0
BEAT_TOL_S = 0.15
SHUFFLE_BLOCK = 60          # 2 s blocks


def head_speed(motion: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    """|angular velocity| of the head (deg/s), one value per frame transition."""
    m = np.asarray(motion, np.float64)[:, HEAD_IDX]
    return np.linalg.norm(np.diff(m, axis=0), axis=1) * rate_hz


def channel_speeds(motion: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    return np.abs(np.diff(np.asarray(motion, np.float64), axis=0)) * rate_hz


def block_shuffle(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Permute ``block``-sized chunks (keeps the statistics, breaks the alignment)."""
    n = len(x)
    n_blocks = max(1, n // block)
    idx = np.arange(n_blocks)
    if n_blocks > 1:
        # a derangement-ish permutation: keep re-drawing until no block stays put
        for _ in range(20):
            rng.shuffle(idx)
            if not np.any(idx == np.arange(n_blocks)):
                break
    parts = [x[i * block:(i + 1) * block] for i in idx]
    out = np.concatenate(parts, axis=0)
    if len(out) < n:
        out = np.concatenate([out, x[len(out):]], axis=0)
    return out[:n]


def peaks(speed: np.ndarray, prominence: float, distance: int = 4) -> np.ndarray:
    from scipy.signal import find_peaks

    if len(speed) < 3:
        return np.zeros(0, np.int64)
    p, _ = find_peaks(speed, prominence=prominence, distance=distance)
    return p.astype(np.int64)


def beat_alignment(gt_speed: np.ndarray, gen_speed: np.ndarray, prominence: float, rate_hz: float = RATE_HZ,
                   tol_s: float = BEAT_TOL_S) -> Dict[str, float]:
    pg = peaks(gt_speed, prominence)
    pm = peaks(gen_speed, prominence)
    tol = tol_s * rate_hz
    if len(pg) == 0 or len(pm) == 0:
        return {"n_gt_peaks": int(len(pg)), "n_gen_peaks": int(len(pm)), "recall": 0.0, "precision": 0.0, "f1": 0.0}
    d = np.abs(pg[:, None] - pm[None, :])
    recall = float((d.min(axis=1) <= tol).mean())
    precision = float((d.min(axis=0) <= tol).mean())
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    return {"n_gt_peaks": int(len(pg)), "n_gen_peaks": int(len(pm)), "recall": recall, "precision": precision, "f1": f1}


def wasserstein_per_channel(gen: np.ndarray, gt: np.ndarray) -> np.ndarray:
    from scipy.stats import wasserstein_distance

    vg, vt = channel_speeds(gen), channel_speeds(gt)
    return np.array([wasserstein_distance(vg[:, i], vt[:, i]) for i in range(vg.shape[1])])


def speed_violations(joints, profile: Profile) -> Dict[str, float]:
    t = joints["t"].to_numpy(dtype=np.float64)
    dt = np.diff(t)
    n_viol = 0
    worst = 0.0
    for j in profile.joints:
        v = np.abs(np.diff(joints[j.name].to_numpy(dtype=np.float64))) / np.maximum(dt, 1e-6)
        ratio = v / j.max_speed
        worst = max(worst, float(ratio.max()) if len(ratio) else 0.0)
        n_viol += int((ratio > 1.0 + 1e-3).sum())
    return {"violations": n_viol, "worst_speed_ratio": worst, "duration_s": float(t[-1] - t[0]) if len(t) else 0.0}


def evaluate(model: MotionModel, index: Optional[RetrievalIndex], val_clips: Sequence[ClipData],
             profiles: Sequence[Profile], seed: int = 0, temperature: float = 0.8, bigram_weight: float = 0.5,
             max_runs: Optional[int] = None, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    stats = model.vq.stats
    unigram = model.info.get("unigram")
    unigram = np.asarray(unigram, np.float64) if unigram is not None else None

    runs = [(c, a, b) for c in val_clips if c.has_audio for a, b in c.runs]
    if max_runs:
        runs = runs[:max_runs]
    if not runs:
        return {"error": "no held-out runs with audio"}

    conds = ["model", "model_shuffled", "model_causal", "retrieval", "retrieval_shuffled"]
    gen: Dict[str, List[np.ndarray]] = {k: [] for k in conds}
    gts: List[np.ndarray] = []
    spk: List[np.ndarray] = []
    nll = {"model": [], "model_causal": [], "unigram": [], "retrieval_eps": []}
    acc = {"model": [], "model_causal": [], "majority": [], "retrieval": []}
    n_steps = 0
    for k, (c, a, b) in enumerate(runs):
        n = ((b - a) // FRAMES_PER_CODE) * FRAMES_PER_CODE
        f, s, gt = c.features[a:a + n], c.speaking[a:a + n], c.motion[a:a + n]
        fs = block_shuffle(f, SHUFFLE_BLOCK, rng)
        ss = block_shuffle(s, SHUFFLE_BLOCK, rng)
        m_model, _ = generate_motion(model, f, s, causal=False, temperature=temperature, bigram_weight=bigram_weight, seed=seed + k)
        m_shuf, _ = generate_motion(model, fs, ss, causal=False, temperature=temperature, bigram_weight=bigram_weight, seed=seed + k)
        m_causal, _ = generate_motion(model, f, s, causal=True, temperature=temperature, bigram_weight=bigram_weight, seed=seed + k)
        gen["model"].append(m_model)
        gen["model_shuffled"].append(m_shuf)
        gen["model_causal"].append(m_causal)
        if index is not None and len(index):
            gen["retrieval"].append(index.query(f, s))
            gen["retrieval_shuffled"].append(index.query(fs, ss))
        gts.append(gt)
        spk.append(s)

        # --- code-level metrics on this run
        codes_gt = model.vq.encode(normalise(gt, stats))
        f15, s15 = pool_pairs(f), pool_flag(s)
        for name, causal in (("model", False), ("model_causal", True)):
            lg = model.a2m.logits(f15, s15, causal=causal).astype(np.float64)
            lg = lg - lg.max(axis=1, keepdims=True)
            logp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
            nll[name].append(-logp[np.arange(len(codes_gt)), codes_gt])
            acc[name].append((lg.argmax(axis=1) == codes_gt).astype(np.float64))
        if unigram is not None:
            nll["unigram"].append(-np.log(np.maximum(unigram[codes_gt], 1e-12)))
            acc["majority"].append((codes_gt == int(unigram.argmax())).astype(np.float64))
        if gen["retrieval"]:
            codes_r = model.vq.encode(normalise(gen["retrieval"][-1], stats))
            eps, nc = 0.05, model.n_codes
            hit = codes_r[:len(codes_gt)] == codes_gt
            acc["retrieval"].append(hit.astype(np.float64))
            nll["retrieval_eps"].append(-np.log(np.where(hit, 1 - eps + eps / nc, eps / nc)))
        n_steps += len(codes_gt)

    def cat(xs):
        return np.concatenate(xs) if xs else np.zeros(0)

    out: Dict = {
        "held_out": {"n_clips": len({c.name for c, _, _ in runs}), "n_runs": len(runs),
                     "frames": int(sum(len(g) for g in gts)), "seconds": round(sum(len(g) for g in gts) / RATE_HZ, 1),
                     "code_steps": int(n_steps)},
        "codes": {
            "nll_model": float(cat(nll["model"]).mean()),
            "nll_model_causal": float(cat(nll["model_causal"]).mean()),
            "nll_unigram_floor": float(cat(nll["unigram"]).mean()) if nll["unigram"] else None,
            "nll_retrieval_eps0.05": float(cat(nll["retrieval_eps"]).mean()) if nll["retrieval_eps"] else None,
            "top1_model": float(cat(acc["model"]).mean()),
            "top1_model_causal": float(cat(acc["model_causal"]).mean()),
            "top1_majority_floor": float(cat(acc["majority"]).mean()) if acc["majority"] else None,
            "top1_retrieval": float(cat(acc["retrieval"]).mean()) if acc["retrieval"] else None,
        },
    }

    # --- motion statistics
    gt_all = np.concatenate(gts)
    gt_speed_all = np.concatenate([head_speed(g) for g in gts])
    prominence = max(3.0, 0.5 * float(gt_speed_all.std()))
    out["beat"] = {"prominence_deg_per_s": prominence, "tolerance_s": BEAT_TOL_S}
    out["velocity"] = {"channels": list(MODEL_CHANNELS), "gt_mean_speed": channel_speeds(gt_all).mean(axis=0).round(4).tolist()}
    out["stillness"] = {"threshold_deg_per_s": STILL_DEG_PER_S, "gt": float((gt_speed_all < STILL_DEG_PER_S).mean())}
    for cond in conds:
        if not gen[cond]:
            continue
        g_all = np.concatenate(gen[cond])
        w = wasserstein_per_channel(g_all, gt_all)
        out["velocity"][cond] = {
            "w1_per_channel": w.round(4).tolist(),
            "w1_mean": float(w.mean()),
            "w1_relative_mean": float(np.mean(w / np.maximum(channel_speeds(gt_all).mean(axis=0), 1e-6))),
            "mean_speed": channel_speeds(g_all).mean(axis=0).round(4).tolist(),
        }
        sp = np.concatenate([head_speed(g) for g in gen[cond]])
        out["stillness"][cond] = float((sp < STILL_DEG_PER_S).mean())
        ba = [beat_alignment(head_speed(gt), head_speed(g), prominence) for gt, g in zip(gts, gen[cond])]
        n_gt = sum(x["n_gt_peaks"] for x in ba)
        n_gen = sum(x["n_gen_peaks"] for x in ba)
        rec = sum(x["recall"] * x["n_gt_peaks"] for x in ba) / max(n_gt, 1)
        prec = sum(x["precision"] * x["n_gen_peaks"] for x in ba) / max(n_gen, 1)
        out["beat"][cond] = {"n_gt_peaks": n_gt, "n_gen_peaks": n_gen, "recall": rec, "precision": prec,
                             "f1": 2 * rec * prec / max(rec + prec, 1e-9)}

    # --- retarget legality on every generated clip
    out["legality"] = {}
    for prof in profiles:
        for cond in ("model", "model_causal", "retrieval"):
            if not gen[cond]:
                continue
            viol, worst, stretch = 0, 0.0, []
            for g, s in zip(gen[cond], spk):
                clip = motion_to_clip(g, s)
                joints = retarget_clip(clip, prof)
                r = speed_violations(joints, prof)
                viol += r["violations"]
                worst = max(worst, r["worst_speed_ratio"])
                stretch.append(r["duration_s"] / max(clip.duration, 1e-6))
            out["legality"][f"{prof.name}/{cond}"] = {"violations": viol, "worst_speed_ratio": round(worst, 4),
                                                       "mean_time_stretch": round(float(np.mean(stretch)), 4)}

    # --- verdict: the learned model must beat its own shuffle on beat recall
    b = out["beat"]
    beats_shuffle = b.get("model", {}).get("recall", 0.0) > b.get("model_shuffled", {}).get("recall", 0.0)
    out["verdict"] = {
        "model_beats_shuffled_audio_on_beat_recall": bool(beats_shuffle),
        "model_beat_recall": b.get("model", {}).get("recall"),
        "model_shuffled_beat_recall": b.get("model_shuffled", {}).get("recall"),
        "retrieval_beat_recall": b.get("retrieval", {}).get("recall"),
        "retrieval_shuffled_beat_recall": b.get("retrieval_shuffled", {}).get("recall"),
        "default_backend": "model" if beats_shuffle else "retrieval",
    }
    if verbose:
        c = out["codes"]
        print(f"  codes: NLL model {c['nll_model']:.3f} / causal {c['nll_model_causal']:.3f} / unigram floor "
              f"{c['nll_unigram_floor'] if c['nll_unigram_floor'] is None else round(c['nll_unigram_floor'], 3)} ; "
              f"top1 model {c['top1_model']:.3f} / majority {c['top1_majority_floor']} / retrieval {c['top1_retrieval']}")
        for cond in conds:
            if cond in out["beat"]:
                bb, st, vv = out["beat"][cond], out["stillness"][cond], out["velocity"][cond]
                print(f"  {cond:18s} beat recall {bb['recall']:.3f} prec {bb['precision']:.3f} (gt {bb['n_gt_peaks']} / gen {bb['n_gen_peaks']} peaks) "
                      f"still {st:.3f} (gt {out['stillness']['gt']:.3f})  W1 rel {vv['w1_relative_mean']:.3f}")
        for k, v in out["legality"].items():
            print(f"  legality {k}: {v['violations']} violations, worst ratio {v['worst_speed_ratio']}, stretch x{v['mean_time_stretch']}")
        print(f"  verdict: {'model beats shuffled audio' if beats_shuffle else 'MODEL DOES NOT BEAT SHUFFLED AUDIO'} -> default backend = {out['verdict']['default_backend']}")
    return out
