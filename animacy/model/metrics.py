"""Held-out evaluation, exactly the list in ``docs/MODEL.md``:

* code NLL / top-1 vs a unigram ("majority code") floor and vs retrieval;
* motion statistics: per-channel |velocity| histogram distance (Wasserstein-1)
  and stillness ratio vs ground truth, for every generator and its
  shuffled-audio control;
* beat alignment: fraction of ground-truth head-velocity peaks within +-150 ms
  of a generated peak, and the same with block-shuffled audio (a generator
  must beat its own shuffle);
* retarget legality: speed-cap violations after ``retarget_clip`` on every
  robot profile given.

Generators: ``model`` = the feed-forward audio -> codes model ("ff"),
``ar`` = the autoregressive one, ``retrieval`` = motion matching. Everything
is computed on whole contiguous runs of held-out clips.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..profile import Profile
from ..retarget import retarget_clip
from ..schema import RATE_HZ
from .data import FRAMES_PER_CODE, MODEL_CHANNELS, ClipData, normalise, pool_flag, pool_pairs
from .infer import MotionModel, generate_motion, motion_to_clip, postprocess_motion, smooth_motion
from .retrieval import RetrievalIndex

HEAD_IDX = [MODEL_CHANNELS.index(c) for c in ("head_yaw", "head_pitch", "head_roll")]
STILL_DEG_PER_S = 5.0
BEAT_TOL_S = 0.15
BEAT_MIN_DISTANCE = 6       # frames (200 ms) between counted peaks
BEAT_MIN_PROMINENCE = 10.0  # deg/s
SHUFFLE_BLOCK = 60          # 2 s blocks
METRIC_SMOOTH_HZ = 6.0      # every condition, ground truth included, is filtered the same way
                            # before velocities are taken: tracking jitter is not motion
BEAT_MARGIN = 0.05          # recall points a generator must beat its own shuffle by; the shuffle
                            # itself moves by ~+-0.04 across sampling settings on 3 min of held-out data
ARCH_COND = {"ff": "model", "ar": "ar"}     # arch -> condition name in the tables


def head_speed(motion: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    """|angular velocity| of the (smoothed) head (deg/s), one value per frame transition."""
    m = smooth_motion(np.asarray(motion, np.float64), rate_hz, METRIC_SMOOTH_HZ)[:, HEAD_IDX]
    return np.linalg.norm(np.diff(m, axis=0), axis=1) * rate_hz


def all_channel_speed(motion: np.ndarray, std: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Speed over all 14 channels in standardised units (sd/s): brows and mouth count too."""
    m = smooth_motion(np.asarray(motion, np.float64), rate_hz, METRIC_SMOOTH_HZ) / np.asarray(std, np.float64)
    return np.linalg.norm(np.diff(m, axis=0), axis=1) * rate_hz


def channel_speeds(motion: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    m = smooth_motion(np.asarray(motion, np.float64), rate_hz, METRIC_SMOOTH_HZ)
    return np.abs(np.diff(m, axis=0)) * rate_hz


def block_shuffle(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Permute ``block``-sized chunks (keeps the statistics, breaks the alignment)."""
    n = len(x)
    n_blocks = max(1, n // block)
    idx = np.arange(n_blocks)
    if n_blocks > 1:
        for _ in range(20):                       # a derangement-ish permutation
            rng.shuffle(idx)
            if not np.any(idx == np.arange(n_blocks)):
                break
    parts = [x[i * block:(i + 1) * block] for i in idx]
    out = np.concatenate(parts, axis=0)
    if len(out) < n:
        out = np.concatenate([out, x[len(out):]], axis=0)
    return out[:n]


def peaks(speed: np.ndarray, prominence: float, distance: int = BEAT_MIN_DISTANCE) -> np.ndarray:
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


def _pooled_beats(ba: List[Dict]) -> Dict[str, float]:
    n_gt = sum(x["n_gt_peaks"] for x in ba)
    n_gen = sum(x["n_gen_peaks"] for x in ba)
    rec = sum(x["recall"] * x["n_gt_peaks"] for x in ba) / max(n_gt, 1)
    prec = sum(x["precision"] * x["n_gen_peaks"] for x in ba) / max(n_gen, 1)
    return {"n_gt_peaks": n_gt, "n_gen_peaks": n_gen, "recall": rec, "precision": prec, "f1": 2 * rec * prec / max(rec + prec, 1e-9)}


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


def _log_softmax(lg: np.ndarray) -> np.ndarray:
    lg = np.asarray(lg, np.float64)
    lg = lg - lg.max(axis=1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))


def compute_verdict(out: Dict) -> Dict:
    """Which generator ships as the default. A learned generator qualifies only if it beats
    its own shuffled-audio control on head-beat recall by ``BEAT_MARGIN`` AND its held-out
    code NLL is below the unigram floor; the autoregressive model is preferred, then the
    feed-forward one, else retrieval."""
    b, c = out["beat"], out["codes"]
    floor = c.get("nll_unigram_floor")
    cands = {}
    for arch, cond in ARCH_COND.items():
        if cond not in b:
            continue
        rm, rs = b[cond]["recall"], b[f"{cond}_shuffled"]["recall"]
        nll = c.get(f"nll_{cond}")
        margin = rm - rs
        cands[arch] = {"condition": cond, "beat_recall": rm, "shuffled_beat_recall": rs, "margin": margin,
                       "nll": nll, "nll_unigram_floor": floor,
                       "beats_shuffle": bool(margin >= BEAT_MARGIN),
                       "below_floor": bool(nll is not None and floor is not None and nll < floor),
                       "stillness": out["stillness"].get(cond), "w1_relative_mean": out["velocity"].get(cond, {}).get("w1_relative_mean")}
        cands[arch]["qualifies"] = cands[arch]["beats_shuffle"] and cands[arch]["below_floor"]
    default = "retrieval"
    for arch in ("ar", "ff"):
        if arch in cands and cands[arch]["qualifies"]:
            default = arch if arch == "ar" else "model"
            break
    best_arch = max(cands, key=lambda a: cands[a]["margin"]) if cands else None
    return {
        "required_margin": BEAT_MARGIN,
        "candidates": cands,
        "default_backend": default,
        "default_arch": best_arch,
        "retrieval_beat_recall": b.get("retrieval", {}).get("recall"),
        "retrieval_shuffled_beat_recall": b.get("retrieval_shuffled", {}).get("recall"),
        # kept for readers of the v1 metrics.json
        "model_beats_shuffled_audio_on_beat_recall": bool(cands.get("ff", {}).get("beats_shuffle", False)),
        "model_beat_recall": cands.get("ff", {}).get("beat_recall"),
        "model_shuffled_beat_recall": cands.get("ff", {}).get("shuffled_beat_recall"),
        "margin": cands.get("ff", {}).get("margin"),
        "nll_model_vs_unigram_floor": [c.get("nll_model"), floor],
    }


STILL_TOL = 0.05            # promotion: generated stillness within this of the ground truth


def compact(ev: Dict) -> Dict:
    """The per-condition numbers that matter, for per-clip tables."""
    out = {"seconds": ev["held_out"]["seconds"], "gt_stillness": ev["stillness"]["gt"],
           "nll_unigram_floor": ev["codes"].get("nll_unigram_floor"), "conditions": {}}
    for cond in ev.get("conditions", []):
        if cond in ev["beat"]:
            out["conditions"][cond] = {"beat_recall": ev["beat"][cond]["recall"], "beat_precision": ev["beat"][cond]["precision"],
                                       "stillness": ev["stillness"][cond], "w1_relative_mean": ev["velocity"][cond]["w1_relative_mean"],
                                       "nll": ev["codes"].get(f"nll_{cond}")}
    return out


def promotion_verdict(per_clip: Dict[str, Dict], floor_key: str = "nll_unigram_floor") -> Dict:
    """The shipping rule: a learned generator becomes the default only if on EVERY held-out
    speaker it beats its shuffled-audio control by >= BEAT_MARGIN, sits within STILL_TOL of the
    ground-truth stillness, and its code NLL is below the unigram floor. AR preferred over ff."""
    out = {"rule": f"on every held-out clip: margin >= {BEAT_MARGIN}, |stillness - gt| <= {STILL_TOL}, NLL < unigram floor",
           "candidates": {}, "default_backend": "retrieval"}
    for arch, cond in ARCH_COND.items():
        rows = {}
        ok_all = True
        for name, pc in per_clip.items():
            c = pc["conditions"].get(cond)
            cs = pc["conditions"].get(f"{cond}_shuffled")
            if c is None or cs is None:
                ok_all = False
                continue
            margin = c["beat_recall"] - cs["beat_recall"]
            still_gap = abs(c["stillness"] - pc["gt_stillness"])
            below = c["nll"] is not None and pc[floor_key] is not None and c["nll"] < pc[floor_key]
            ok = margin >= BEAT_MARGIN and still_gap <= STILL_TOL and below
            rows[name] = {"margin": margin, "stillness": c["stillness"], "gt_stillness": pc["gt_stillness"],
                          "still_gap": still_gap, "nll": c["nll"], "floor": pc[floor_key], "below_floor": bool(below), "ok": bool(ok)}
            ok_all &= ok
        if rows:
            out["candidates"][arch] = {"per_clip": rows, "qualifies": bool(ok_all and len(rows) == len(per_clip))}
    for arch in ("ar", "ff"):
        if out["candidates"].get(arch, {}).get("qualifies"):
            out["default_backend"] = "ar" if arch == "ar" else "model"
            break
    return out


def postprocess_rows(model: MotionModel, index: Optional[RetrievalIndex], val_clips: Sequence[ClipData],
                     sampling: Dict, archs: Optional[Sequence[str]] = None, seed: int = 0,
                     grid=((0.0, None, 1.0, False), (0.5, -3.0, 1.0, False), (0.0, None, 1.2, False), (0.5, -3.0, 1.2, False),
                           (0.5, -3.0, 1.0, True))) -> List[Dict]:
    """The generation-side options as separate rows, per held-out clip, at the shipped sampling:
    (settle_s, pitch_floor, amplitude, intent_from_audio) for every learned generator and retrieval.
    The last row is the intent layer driven by the clip's own audio (no text): retrieval arousal
    bonus + amplitude rule."""
    archs = list(archs) if archs is not None else model.archs
    conds = [ARCH_COND[a] for a in archs] + (["retrieval"] if index is not None and len(index) else [])
    rows = []
    for settle_s, pitch_floor, amp, ifa in grid:
        if ifa and (index is None or index.arousal is None):
            continue
        per = {}
        arous = {}
        for c in val_clips:
            e = evaluate(model, index, [c], [], seed=seed, temperature=sampling.get("temperature", 1.0),
                         bigram_weight=sampling.get("bigram_weight", 0.5), top_p=sampling.get("top_p", 1.0),
                         repeat_penalty=sampling.get("repeat_penalty", 0.0), stay_bias=sampling.get("stay_bias", 0.0),
                         stay_energy=sampling.get("stay_energy", -0.3), settle_s=settle_s, pitch_floor=pitch_floor,
                         amplitude=amp, archs=archs, verbose=False, intent_from_audio=ifa)
            per[c.name] = compact(e)
            arous[c.name] = e["postprocess"].get("audio_arousal_mean")
        row = {"settle_s": settle_s, "pitch_floor": pitch_floor, "amplitude": amp, "intent_from_audio": ifa,
               "audio_arousal": arous if ifa else None, "conditions": {}}
        for cond in conds:
            vals = [pc["conditions"][cond] for pc in per.values() if cond in pc["conditions"]]
            shuf = [pc["conditions"][f"{cond}_shuffled"] for pc in per.values() if f"{cond}_shuffled" in pc["conditions"]]
            if not vals:
                continue
            n = len(vals)
            row["conditions"][cond] = {
                "beat_recall": sum(v["beat_recall"] for v in vals) / n,
                "beat_recall_shuffled": sum(v["beat_recall"] for v in shuf) / n,
                "margin_min": min(v["beat_recall"] - s["beat_recall"] for v, s in zip(vals, shuf)),
                "stillness": sum(v["stillness"] for v in vals) / n,
                "gt_stillness": sum(pc["gt_stillness"] for pc in per.values()) / n,
                "w1_relative_mean": sum(v["w1_relative_mean"] for v in vals) / n,
                "per_clip": {name: {"margin": pc["conditions"][cond]["beat_recall"] - pc["conditions"][f"{cond}_shuffled"]["beat_recall"],
                                    "stillness": pc["conditions"][cond]["stillness"], "w1": pc["conditions"][cond]["w1_relative_mean"]}
                             for name, pc in per.items() if cond in pc["conditions"]},
            }
        rows.append(row)
    return rows


def evaluate(model: MotionModel, index: Optional[RetrievalIndex], val_clips: Sequence[ClipData],
             profiles: Sequence[Profile], seed: int = 0, temperature: float = 0.8, bigram_weight: float = 0.5,
             top_p: float = 0.9, archs: Optional[Sequence[str]] = None, max_runs: Optional[int] = None,
             verbose: bool = True, repeat_penalty: float = 0.0, stay_bias: float = 0.0, stay_energy: float = -0.3,
             settle_s: float = 0.0, pitch_floor: Optional[float] = None, amplitude=1.0,
             intent_from_audio: bool = False, proto_weight: float = 0.0, energy_floor: Optional[float] = None,
             gesture_placement=None) -> Dict:
    """``intent_from_audio``: the intent layer with no text - retrieval uses the query window's
    own audio arousal for its bonus, and every source's amplitude follows the amplitude rule on
    the run's mean audio arousal (what ``animacy say`` would do without a text intent).
    ``proto_weight``: gesture-prototype bonus in retrieval with a pseudo-intent from the run's
    audio arousal (> 0.6 excitement, < 0.3 thinking, else agreement) - the held-out clips have
    no text, this is the closest honest proxy. ``energy_floor``: per-utterance floor on every source."""
    rng = np.random.default_rng(seed)
    stats = model.vq.stats
    unigram = model.info.get("unigram")
    unigram = np.asarray(unigram, np.float64) if unigram is not None else None
    archs = list(archs) if archs is not None else model.archs

    runs = [(c, a, b) for c in val_clips if c.has_audio for a, b in c.runs]
    if max_runs:
        runs = runs[:max_runs]
    if not runs:
        return {"error": "no held-out runs with audio"}

    conds: List[str] = []
    for arch in archs:
        cn = ARCH_COND[arch]
        conds += [cn, f"{cn}_shuffled", f"{cn}_causal"]
    if index is not None and len(index):
        conds += ["retrieval", "retrieval_shuffled"]
    gen: Dict[str, List[np.ndarray]] = {k: [] for k in conds}
    gts: List[np.ndarray] = []
    spk: List[np.ndarray] = []
    audio_arousal_runs: List[float] = []
    nll: Dict[str, List[np.ndarray]] = {}
    acc: Dict[str, List[np.ndarray]] = {}
    n_steps = 0
    for k, (c, a, b) in enumerate(runs):
        n = ((b - a) // FRAMES_PER_CODE) * FRAMES_PER_CODE
        f, s, gt = c.features[a:a + n], c.speaking[a:a + n], c.motion[a:a + n]
        fs = block_shuffle(f, SHUFFLE_BLOCK, rng)
        ss = block_shuffle(s, SHUFFLE_BLOCK, rng)
        amp, amp_s, use_aa = amplitude, amplitude, False
        tag_run, tag_shuf = None, None
        if (intent_from_audio or proto_weight > 0) and index is not None and len(index) and index.arousal is not None:
            from .intent import amplitude_for

            win = int(index.meta.get("win", 30))
            ar_run = float(np.mean([index.audio_arousal(f[i:i + win]) for i in range(0, max(1, n - win + 1), win)]))
            ar_shuf = float(np.mean([index.audio_arousal(fs[i:i + win]) for i in range(0, max(1, n - win + 1), win)]))
            audio_arousal_runs.append(ar_run)
            if intent_from_audio:
                amp, amp_s, use_aa = amplitude_for(ar_run), amplitude_for(ar_shuf), True
            if proto_weight > 0:
                pseudo = lambda a: "excitement" if a > 0.6 else ("thinking" if a < 0.3 else "agreement")  # noqa: E731
                tag_run, tag_shuf = pseudo(ar_run), pseudo(ar_shuf)
        pp = dict(settle_s=settle_s, pitch_floor=pitch_floor, amplitude=amp, energy_floor=energy_floor, energy_stats=stats)
        pp_s = dict(settle_s=settle_s, pitch_floor=pitch_floor, amplitude=amp_s, energy_floor=energy_floor, energy_stats=stats)
        kw = dict(temperature=temperature, bigram_weight=bigram_weight, top_p=top_p, seed=seed + k, repeat_penalty=repeat_penalty,
                  stay_bias=stay_bias, stay_energy=stay_energy, settle_s=settle_s, pitch_floor=pitch_floor, energy_floor=energy_floor)
        for arch in archs:
            cn = ARCH_COND[arch]
            gen[cn].append(generate_motion(model, f, s, causal=False, arch=arch, amplitude=amp, **kw)[0])
            gen[f"{cn}_shuffled"].append(generate_motion(model, fs, ss, causal=False, arch=arch, amplitude=amp_s, **kw)[0])
            gen[f"{cn}_causal"].append(generate_motion(model, f, s, causal=True, arch=arch, amplitude=amp, **kw)[0])
        if "retrieval" in gen:
            r_run = index.query(f, s, use_audio_arousal=use_aa, intent_tag=tag_run, proto_weight=proto_weight)
            r_shuf = index.query(fs, ss, use_audio_arousal=use_aa, intent_tag=tag_shuf, proto_weight=proto_weight)
            if gesture_placement not in (None, False, 0) and tag_run is not None:
                from .gesture import PlacementConfig, place_gestures
                from .intent import AMPLITUDE_TIERS

                cfg = PlacementConfig.from_any(gesture_placement)
                r_run, _ = place_gestures(r_run, f, s, index, tag_run, AMPLITUDE_TIERS.get(tag_run, 1.0), cfg)
                r_shuf, _ = place_gestures(r_shuf, fs, ss, index, tag_shuf, AMPLITUDE_TIERS.get(tag_shuf, 1.0), cfg)
            gen["retrieval"].append(postprocess_motion(r_run, s, f, **pp))
            gen["retrieval_shuffled"].append(postprocess_motion(r_shuf, ss, fs, **pp_s))
        gts.append(gt)
        spk.append(s)

        # --- code-level metrics on this run
        codes_gt = model.vq.encode(normalise(gt, stats))
        f15, s15 = pool_pairs(f), pool_flag(s)
        for arch in archs:
            cn = ARCH_COND[arch]
            for name, causal in ((cn, False), (f"{cn}_causal", True)):
                if arch == "ff":
                    lg = model.a2m.logits(f15, s15, causal=causal)
                else:
                    lg = model.ar.teacher_forced_logits(f15, s15, codes_gt, causal=causal)
                logp = _log_softmax(lg)
                nll.setdefault(name, []).append(-logp[np.arange(len(codes_gt)), codes_gt])
                acc.setdefault(name, []).append((logp.argmax(axis=1) == codes_gt).astype(np.float64))
        if unigram is not None:
            nll.setdefault("unigram", []).append(-np.log(np.maximum(unigram[codes_gt], 1e-12)))
            acc.setdefault("majority", []).append((codes_gt == int(unigram.argmax())).astype(np.float64))
        if "retrieval" in gen:
            codes_r = model.vq.encode(normalise(gen["retrieval"][-1], stats))
            eps, nc = 0.05, model.n_codes
            hit = codes_r[:len(codes_gt)] == codes_gt
            acc.setdefault("retrieval", []).append(hit.astype(np.float64))
            nll.setdefault("retrieval_eps", []).append(-np.log(np.where(hit, 1 - eps + eps / nc, eps / nc)))
        n_steps += len(codes_gt)

    def mean_of(d, key):
        return float(np.concatenate(d[key]).mean()) if key in d and d[key] else None

    codes_out = {
        "nll_unigram_floor": mean_of(nll, "unigram"),
        "top1_majority_floor": mean_of(acc, "majority"),
        "nll_retrieval_eps0.05": mean_of(nll, "retrieval_eps"),
        "top1_retrieval": mean_of(acc, "retrieval"),
    }
    for arch in archs:
        cn = ARCH_COND[arch]
        codes_out[f"nll_{cn}"] = mean_of(nll, cn)
        codes_out[f"nll_{cn}_causal"] = mean_of(nll, f"{cn}_causal")
        codes_out[f"top1_{cn}"] = mean_of(acc, cn)
        codes_out[f"top1_{cn}_causal"] = mean_of(acc, f"{cn}_causal")
    out: Dict = {
        "held_out": {"n_clips": len({c.name for c, _, _ in runs}), "n_runs": len(runs),
                     "frames": int(sum(len(g) for g in gts)), "seconds": round(sum(len(g) for g in gts) / RATE_HZ, 1),
                     "code_steps": int(n_steps)},
        "archs": archs,
        "conditions": conds,
        "sampling": {"temperature": temperature, "bigram_weight": bigram_weight, "top_p": top_p, "repeat_penalty": repeat_penalty,
                     "stay_bias": stay_bias, "stay_energy": stay_energy},
        "postprocess": {"settle_s": settle_s, "pitch_floor": pitch_floor,
                        "amplitude": (float(amplitude) if np.isscalar(amplitude) else [float(x) for x in amplitude]),
                        "intent_from_audio": intent_from_audio, "proto_weight": proto_weight, "energy_floor": energy_floor,
                        "gesture_placement": (None if gesture_placement in (None, False, 0) else True),
                        "audio_arousal_mean": (float(np.mean(audio_arousal_runs)) if audio_arousal_runs else None)},
        "codes": codes_out,
    }

    # --- motion statistics
    gt_all = np.concatenate(gts)
    gt_speed_all = np.concatenate([head_speed(g) for g in gts])
    prominence = max(BEAT_MIN_PROMINENCE, 1.0 * float(gt_speed_all.std()))
    out["beat"] = {"prominence_deg_per_s": prominence, "tolerance_s": BEAT_TOL_S, "min_distance_frames": BEAT_MIN_DISTANCE,
                   "smooth_hz": METRIC_SMOOTH_HZ}
    gt_aspeed_all = np.concatenate([all_channel_speed(g, stats["std"]) for g in gts])
    prominence_all = 1.0 * float(gt_aspeed_all.std())
    out["beat_all_channels"] = {"prominence_sd_per_s": prominence_all, "tolerance_s": BEAT_TOL_S,
                                "note": "secondary: peaks of standardised speed over all 14 channels (brows, mouth included)"}
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
        out["beat"][cond] = _pooled_beats([beat_alignment(head_speed(gt), head_speed(g), prominence) for gt, g in zip(gts, gen[cond])])
        out["beat_all_channels"][cond] = _pooled_beats([beat_alignment(all_channel_speed(gt, stats["std"]), all_channel_speed(g, stats["std"]), prominence_all)
                                                        for gt, g in zip(gts, gen[cond])])

    # --- retarget legality on every generated clip
    out["legality"] = {}
    for prof in profiles:
        for cond in conds:
            if cond.endswith("_shuffled") or not gen[cond]:
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

    out["verdict"] = compute_verdict(out)
    if verbose:
        c = out["codes"]
        line = f"  codes: unigram floor NLL {c['nll_unigram_floor'] if c['nll_unigram_floor'] is None else round(c['nll_unigram_floor'], 3)}"
        for arch in archs:
            cn = ARCH_COND[arch]
            line += f" | {cn}: NLL {c[f'nll_{cn}']:.3f} / causal {c[f'nll_{cn}_causal']:.3f}, top1 {c[f'top1_{cn}']:.3f}"
        print(line + f" | retrieval top1 {c['top1_retrieval']}")
        for cond in conds:
            if cond in out["beat"]:
                bb, st, vv, bA = out["beat"][cond], out["stillness"][cond], out["velocity"][cond], out["beat_all_channels"][cond]
                print(f"  {cond:18s} head-beat recall {bb['recall']:.3f} prec {bb['precision']:.3f} (gt {bb['n_gt_peaks']} / gen {bb['n_gen_peaks']}) "
                      f"all-ch recall {bA['recall']:.3f} prec {bA['precision']:.3f}  still {st:.3f} (gt {out['stillness']['gt']:.3f})  W1 rel {vv['w1_relative_mean']:.3f}")
        for k, v in out["legality"].items():
            print(f"  legality {k}: {v['violations']} violations, worst ratio {v['worst_speed_ratio']}, stretch x{v['mean_time_stretch']}")
        vd = out["verdict"]
        for arch, cd in vd["candidates"].items():
            print(f"  {arch}: margin over shuffle {cd['margin']:+.3f} (need >= {BEAT_MARGIN}), NLL {cd['nll']:.3f} vs floor "
                  f"{cd['nll_unigram_floor'] if cd['nll_unigram_floor'] is None else round(cd['nll_unigram_floor'], 3)} -> "
                  f"{'QUALIFIES' if cd['qualifies'] else 'does not qualify'}")
        print(f"  verdict: default backend = {vd['default_backend']}")
    return out
