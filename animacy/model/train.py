"""Train the motion model: stage 1 VQ tokenizer, stage 2 audio -> codes (feed-forward
``a2m`` and/or autoregressive ``a2m_ar``), then the held-out metrics from
``docs/MODEL.md``, then the browser export.

    python -m animacy.model.train --data data/clips --out checkpoints/v1 --holdout kende_interview_2014
    python -m animacy.model.train --synthetic --out checkpoints/synthetic --epochs-vq 3 --epochs-a2m 3

Writes ``<out>/vq.pt``, ``<out>/a2m.pt``, ``<out>/a2m_ar.pt``, ``<out>/model_info.json``,
``<out>/metrics.json``, ``<out>/REPORT.md`` and (unless ``--no-export``) the ``web/models/``
bundle. Nothing is claimed as measured that is not in metrics.json.

Data refresh / GPU workflow: ``--init-vq checkpoints/v1/vq.pt --epochs-vq 0`` keeps the
tokenizer (and its codes) fixed across versions, ``--init-a2m`` / ``--init-ar`` warm-start
the predictors, audio features are cached under ``--cache-dir``. Train on a CUDA venv
without onnx (``--device cuda``): the export is skipped with a flag and
``python -m animacy.model.export --ckpt <out> --out web/models`` finishes it elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..profile import load_profile, robots_root
from .a2m import AudioToMotion, BigramPrior
from .a2m_ar import AudioToMotionAR
from .data import (CHUNK_FRAMES, MODEL_CHANNELS, SEGMENT, a2m_chunks, apply_speaker_cap, compute_stats, load_clips,
                   load_speaker_index, make_synthetic_clips, mirror_clip, run_code_sequences, split_clips, summarise, vq_segments)
from .infer import MotionModel
from .metrics import ARCH_COND, compact, evaluate, postprocess_rows, promotion_verdict
from .retrieval import RetrievalIndex
from .vq import MotionVQVAE

MIN_USED_CODES_REAL = 200


def _device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _log(fh, msg: str) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------
def train_vq(segs_tr: np.ndarray, segs_va: np.ndarray, stats, args, device: str, log, init: Optional[MotionVQVAE] = None) -> tuple:
    if init is not None:
        model = init.to(device)
    else:
        model = MotionVQVAE(n_codes=args.n_codes, dim=args.vq_dim, revive_after=args.vq_revive_after).to(device)
        model.set_stats(stats)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(segs_tr)
    xv = torch.from_numpy(segs_va).to(device) if len(segs_va) else None
    base = float(np.abs(segs_va).mean()) if len(segs_va) else float("nan")   # predict the (zero) mean
    log(f"  VQ-VAE {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {model.n_codes} codes x {model.config['dim']}d; "
        f"train {len(segs_tr)} / val {len(segs_va)} segments of {SEGMENT} frames; baseline val MAE {base:.4f}"
        + ("; warm start" if init is not None else ""))
    history: List[Dict] = []
    best, best_state = float("inf"), None
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs_vq + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot = 0.0
        for i in range(0, len(xt), args.batch_size):
            xb = xt[perm[i:i + args.batch_size]].to(device)
            recon, commit, _ = model(xb)
            loss = F.mse_loss(recon, xb) + args.commit_weight * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        rec: Dict = {"epoch": epoch, "train_loss": tot / max(len(xt), 1)}
        with torch.no_grad():
            if xv is not None:
                recon, _, idx = model(xv)
                rec["val_mae"] = float((recon - xv).abs().mean())
                rec["val_used_codes"], rec["val_perplexity"] = model.quantizer.usage(idx)
            used_all = []
            for i in range(0, len(xt), 4096):
                _, _, idx = model(xt[i:i + 4096].to(device))
                used_all.append(idx.flatten())
            rec["train_used_codes"], rec["train_perplexity"] = model.quantizer.usage(torch.cat(used_all))
        history.append(rec)
        score = rec.get("val_mae", rec["train_loss"])
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  vq epoch {epoch:3d} train {rec['train_loss']:.5f}  val MAE {rec.get('val_mae', float('nan')):.4f}  "
            f"codes used (train) {rec['train_used_codes']}/{model.n_codes}  perplexity {rec['train_perplexity']:.1f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, base


# ---------------------------------------------------------------------------
# stage 2 (shared pieces)
# ---------------------------------------------------------------------------
def _augment_features(f: torch.Tensor, args, device: str) -> torch.Tensor:
    if args.feat_noise > 0:
        f = f + args.feat_noise * torch.randn_like(f)
    if args.time_mask > 0:
        # SpecAugment-style: blank up to `time_mask` spans of 1-4 steps per chunk
        B, L = f.shape[:2]
        pos = torch.arange(L, device=device)[None, None, :]
        starts = torch.randint(0, L, (B, args.time_mask, 1), device=device)
        widths = torch.randint(1, 5, (B, args.time_mask, 1), device=device)
        on = torch.rand(B, args.time_mask, 1, device=device) < 0.5
        blank = (((pos >= starts) & (pos < starts + widths)) & on).any(dim=1)
        f = f.masked_fill(blank[:, :, None], 0.0)
    return f


def _eval_a2m(model: AudioToMotion, ch: Dict[str, torch.Tensor], causal: bool) -> Dict[str, float]:
    model.eval()
    nll, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ch["codes"]), 256):
            f, s, c, m = (ch[k][i:i + 256] for k in ("features", "speaking", "codes", "mask"))
            lg = model(f, s, causal, key_padding_mask=~m)
            nll += float(F.cross_entropy(lg[m], c[m], reduction="sum"))
            acc += float((lg.argmax(-1)[m] == c[m]).sum())
            n += int(m.sum())
    return {"nll": nll / max(n, 1), "acc": acc / max(n, 1), "n": n}


def _bos_shift(codes: torch.Tensor, bos: int) -> torch.Tensor:
    return torch.cat([torch.full_like(codes[:, :1], bos), codes[:, :-1]], dim=1)


def _eval_ar(model: AudioToMotionAR, ch: Dict[str, torch.Tensor], causal: bool) -> Dict[str, float]:
    model.eval()
    nll, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ch["codes"]), 128):
            f, s, c, m = (ch[k][i:i + 128] for k in ("features", "speaking", "codes", "mask"))
            lg = model(f, s, causal, _bos_shift(c, model.bos), key_padding_mask=~m, code_padding_mask=~m)
            nll += float(F.cross_entropy(lg[m], c[m], reduction="sum"))
            acc += float((lg.argmax(-1)[m] == c[m]).sum())
            n += int(m.sum())
    return {"nll": nll / max(n, 1), "acc": acc / max(n, 1), "n": n}


def _fit(model, kind: str, chunks_tr, chunks_va, args, device: str, log, epochs: int, lr: float, patience: int) -> List[Dict]:
    """Shared loop for the feed-forward ("ff") and autoregressive ("ar") predictors:
    50/50 causal batches, feature noise + time masking, early stopping on held-out NLL."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    tr = {k: torch.from_numpy(v) for k, v in chunks_tr.items()}
    va = {k: torch.from_numpy(v).to(device) for k, v in chunks_va.items()}
    n_tr, n_va = len(tr["codes"]), len(va["codes"])
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    history: List[Dict] = []
    best, best_state, best_epoch, since = float("inf"), None, 0, 0
    bs = args.a2m_batch_size
    ls = args.label_smoothing if kind == "ar" else 0.0
    evalf = _eval_ar if kind == "ar" else _eval_a2m
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=g)
        tot, nb, n_causal = 0.0, 0, 0
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            f = _augment_features(tr["features"][idx].to(device), args, device)
            s, c, m = tr["speaking"][idx].to(device), tr["codes"][idx].to(device), tr["mask"][idx].to(device)
            causal = bool(rng.random() < 0.5)          # one set of weights serves talk and listen
            n_causal += int(causal)
            if kind == "ar":
                c_in = _bos_shift(c, model.bos)
                lg = model(f, s, causal, c_in, key_padding_mask=~m, code_padding_mask=~m)
                # Transitions carry the information; ~half the targets equal the previous code and
                # plain CE is minimised by copying (a frozen sampler). Upweight the steps that change.
                per = F.cross_entropy(lg[m], c[m], label_smoothing=ls, reduction="none")
                w = torch.where(c[m] != c_in[m], args.ar_change_weight, 1.0).to(per.dtype)
                loss = (per * w).sum() / w.sum()
            else:
                lg = model(f, s, causal, key_padding_mask=~m)
                loss = F.cross_entropy(lg[m], c[m], label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        rec: Dict = {"epoch": epoch, "train_loss": tot / max(nb, 1), "causal_batches": n_causal, "batches": nb}
        if n_va:
            rec["val_noncausal"] = evalf(model, va, False)
            rec["val_causal"] = evalf(model, va, True)
            score = 0.5 * (rec["val_noncausal"]["nll"] + rec["val_causal"]["nll"])
        else:
            score = rec["train_loss"]
        history.append(rec)
        vn, vc = rec.get("val_noncausal", {}), rec.get("val_causal", {})
        log(f"  {kind} epoch {epoch:3d} train CE {rec['train_loss']:.4f}  val NLL non-causal {vn.get('nll', float('nan')):.4f} "
            f"(top1 {vn.get('acc', float('nan')):.3f})  causal {vc.get('nll', float('nan')):.4f} (top1 {vc.get('acc', float('nan')):.3f})")
        if score < best - 1e-4:
            best, best_epoch, since = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if patience and since >= patience:
                log(f"  {kind}: early stop at epoch {epoch} (best epoch {best_epoch}, patience {patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _thin(history: List[Dict], every: int = 5, keep=None) -> List[Dict]:
    n = len(history)
    return [r for i, r in enumerate(history) if i == 0 or (i + 1) % every == 0 or i == n - 1 or r.get("epoch") == keep]


def _stage_table(L: List[str], name: str, info: Dict) -> None:
    hist = info["history"]
    best_ep = min(hist, key=lambda r: r.get("val_noncausal", {}).get("nll", float("inf")))["epoch"] if hist else None
    L.append(f"(best held-out epoch {best_ep} is the saved checkpoint; every 5th epoch shown)\n")
    L.append("| epoch | train CE | val NLL (talk) | top1 | val NLL (listen) | top1 |")
    L.append("|---|---|---|---|---|---|")
    for r in _thin(hist, keep=best_ep):
        vn, vc = r.get("val_noncausal", {}), r.get("val_causal", {})
        L.append(f"| {r['epoch']} | {r['train_loss']:.4f} | {_fmt(vn.get('nll'))} | {_fmt(vn.get('acc'))} | {_fmt(vc.get('nll'))} | {_fmt(vc.get('acc'))} |")
    L.append("")


def write_report(path: str, m: Dict, command: str) -> None:
    d, sp, vq = m["data"], m["split"], m["vq"]
    L = []
    L.append("# animacy motion model - training report\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on device `{m['device']}`.\n")
    L.append("## Command\n")
    L.append("```\n" + command + "\n```\n")
    if m.get("flags"):
        L.append("## Flags\n")
        for f in m["flags"]:
            L.append(f"- **{f}**")
        L.append("")
    L.append("## Data\n")
    L.append(f"- clips: {d['n_clips']} ({d['n_with_audio']} with audio), subjects: {d['n_subjects']}, sources: {d['sources']}")
    L.append(f"- minutes: total {d['total_minutes']}, valid (face_valid runs >= 1 s) {d['valid_minutes']}, valid with audio {d['valid_minutes_with_audio']}")
    L.append(f"- split: by {sp['mode']}; held out {sp.get('held_out_clips', sp['held_out_groups'])} "
             f"({sp.get('held_out_valid_seconds', '?')} s valid) -> train {m['n_train_clips']} / val {m['n_val_clips']} clips"
             + ("  **(LEAKY: time split inside a single group)**" if sp.get("leaky") else ""))
    if m.get("excluded"):
        L.append(f"- excluded clips: {m['excluded']}")
    L.append(f"- augmentation (train only): {m.get('augmentation')}")
    L.append("")
    L.append("## Stage 1: VQ tokenizer\n")
    L.append(f"- {vq['n_codes']} codes x {vq['dim']}d, one code per 2 frames, {vq['n_train_segments']} train / {vq['n_val_segments']} val segments of 8 frames"
             + ("  (warm start from " + vq["init"] + ")" if vq.get("init") else ""))
    L.append(f"- best val MAE (standardised units) {_fmt(vq['best_val_mae'], 4)} vs predict-mean baseline {_fmt(vq['baseline_val_mae'], 4)}")
    L.append(f"- codes used on whole training runs: **{vq['used_codes_train']}/{vq['n_codes']}** (perplexity {_fmt(vq['perplexity_train'], 1)}); "
             f"on held-out runs {vq['used_codes_val']} (perplexity {_fmt(vq['perplexity_val'], 1)})")
    L.append("")
    if vq["history"]:
        L.append("| epoch | train loss | val MAE | used (train) | perplexity |")
        L.append("|---|---|---|---|---|")
        for r in _thin(vq["history"]):
            L.append(f"| {r['epoch']} | {r['train_loss']:.5f} | {_fmt(r.get('val_mae'), 4)} | {r['train_used_codes']} | {r['train_perplexity']:.1f} |")
        L.append("")
    if m.get("a2m"):
        a = m["a2m"]
        L.append("## Stage 2a: audio -> codes, feed-forward (`a2m`)\n")
        L.append(f"- Transformer d {a['d_model']}, {a['n_layers']} layers, {a['n_heads']} heads; {a['n_train_chunks']} train / {a['n_val_chunks']} val chunks of {a['chunk_frames']} frames; causal/non-causal 50/50; {sum(p['batches'] for p in a['history'])} batches")
        L.append("")
        _stage_table(L, "a2m", a)
    if m.get("ar"):
        a = m["ar"]
        L.append("## Stage 2b: audio -> codes, autoregressive (`a2m_ar`)\n")
        L.append(f"- audio trunk d {a['d_model']}, {a['enc_layers']} layers + code decoder {a['dec_layers']} layers (self-attention window {a['window']} codes, cross-attention to audio), "
                 f"{a['n_heads']} heads, dropout {a['dropout']}, label smoothing {a['label_smoothing']}; {a['n_train_chunks']} train / {a['n_val_chunks']} val chunks of {a['chunk_frames']} frames; early stopping patience {a['patience']}")
        L.append("")
        _stage_table(L, "ar", a)
    ev = m.get("eval", {})
    if "codes" in ev:
        c, ho, archs = ev["codes"], ev["held_out"], ev.get("archs", ["ff"])
        L.append("## Held-out metrics\n")
        L.append(f"Held out: {ho['n_clips']} clips, {ho['n_runs']} runs, {ho['seconds']} s, {ho['code_steps']} code steps. "
                 f"Sampling: {ev.get('sampling')}.\n")
        L.append("### Codes (lower NLL / higher top-1 is better; AR NLL is teacher-forced)\n")
        L.append("| predictor | NLL | top-1 |")
        L.append("|---|---|---|")
        L.append(f"| unigram / majority floor | {_fmt(c['nll_unigram_floor'])} | {_fmt(c['top1_majority_floor'])} |")
        L.append(f"| retrieval (eps-smoothed 0.05) | {_fmt(c.get('nll_retrieval_eps0.05'))} | {_fmt(c.get('top1_retrieval'))} |")
        for arch in archs:
            cn = ARCH_COND[arch]
            L.append(f"| {cn}, talk (non-causal) | {_fmt(c.get(f'nll_{cn}'))} | {_fmt(c.get(f'top1_{cn}'))} |")
            L.append(f"| {cn}, listen (causal) | {_fmt(c.get(f'nll_{cn}_causal'))} | {_fmt(c.get(f'top1_{cn}_causal'))} |")
        L.append("")
        L.append("### Motion statistics vs ground truth\n")
        L.append(f"Beat = head |angular velocity| peaks (prominence {ev['beat']['prominence_deg_per_s']:.1f} deg/s) within +-150 ms. "
                 f"Stillness = fraction of frames with head speed < {ev['stillness']['threshold_deg_per_s']} deg/s (ground truth {ev['stillness']['gt']:.3f}). "
                 "W1 = mean over channels of Wasserstein-1 between |velocity| histograms, relative to the ground-truth mean speed.\n")
        L.append("| condition | head-beat recall | head-beat precision | gt/gen peaks | all-channel beat recall | all-ch precision | stillness | W1 rel |")
        L.append("|---|---|---|---|---|---|---|---|")
        for cond in ev.get("conditions", []):
            if cond in ev["beat"]:
                b, v, bA = ev["beat"][cond], ev["velocity"][cond], ev["beat_all_channels"][cond]
                L.append(f"| {cond} | {b['recall']:.3f} | {b['precision']:.3f} | {b['n_gt_peaks']}/{b['n_gen_peaks']} | "
                         f"{bA['recall']:.3f} | {bA['precision']:.3f} | {ev['stillness'][cond]:.3f} | {v['w1_relative_mean']:.3f} |")
        L.append("")
        L.append("`model` = feed-forward a2m, `ar` = autoregressive a2m_ar. The verdict uses the head-beat recall (docs/MODEL.md). "
                 "The all-channel column counts brows and mouth too.\n")
        if ev.get("per_clip") and len(ev["per_clip"]) > 1:
            L.append("### Per held-out speaker (defaults; the promotion rule is applied here)\n")
            L.append("| clip | condition | head-beat recall | vs shuffled | margin | stillness (gt) | W1 rel | NLL (floor) |")
            L.append("|---|---|---|---|---|---|---|---|")
            for name, pc in ev["per_clip"].items():
                for cond in ("model", "ar", "retrieval"):
                    if cond in pc["conditions"]:
                        x, xs = pc["conditions"][cond], pc["conditions"][f"{cond}_shuffled"]
                        L.append(f"| {name} | {cond} | {x['beat_recall']:.3f} | {xs['beat_recall']:.3f} | {x['beat_recall'] - xs['beat_recall']:+.3f} | "
                                 f"{x['stillness']:.3f} ({pc['gt_stillness']:.3f}) | {x['w1_relative_mean']:.3f} | "
                                 f"{_fmt(x['nll'])} ({_fmt(pc['nll_unigram_floor'])}) |")
            L.append("")
        if m.get("sampling_sweep"):
            L.append("### Sampling sweep (per held-out speaker; the defaults above are the reported numbers)\n")
            L.append("| arch | temperature | bigram weight / top-p / rp / stay | mean head-beat recall | vs shuffled | mean margin | min margin | mean stillness | max stillness gap | W1 rel | per clip (margin / stillness) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in m["sampling_sweep"]:
                per = "; ".join(f"{k[:14]}: {v['head_beat_recall'] - v['head_beat_recall_shuffled']:+.3f} / {v['stillness']:.3f}" for k, v in r.get("per_clip", {}).items())
                L.append(f"| {r['arch']} | {r['temperature']} | {r['second']} | {r['head_beat_recall']:.3f} | {r['head_beat_recall_shuffled']:.3f} | "
                         f"{r.get('mean_margin', r['head_beat_recall'] - r['head_beat_recall_shuffled']):+.3f} | {_fmt(r.get('min_margin'))} | {r['stillness']:.3f} | "
                         f"{_fmt(r.get('max_still_gap'))} | {r['w1_relative_mean']:.3f} | {per} |")
            L.append("")
            smp = m.get("sampling") or {}
            if smp.get("picked_by"):
                L.append(f"Shipped AR sampling: T={smp['temperature']} top_p={smp['top_p']} repeat_penalty={smp['repeat_penalty']} "
                         f"stay_bias={smp['stay_bias']} ({smp['picked_by']}).\n")
        if m.get("postprocess_rows"):
            L.append("### Generation-side options (utterance-final settle, head_pitch floor, amplitude), mean over held-out speakers\n")
            L.append("Applied after decoding to every source. Shipped defaults: " + str(m.get("postprocess")) + ". "
                     "Beat/stillness/W1 as above; `min margin` is the worse of the two speakers.\n")
            L.append("| settle (s) | pitch floor (deg) | amplitude | source | beat recall | vs shuffled | min margin | stillness (gt) | W1 rel | per clip (margin / stillness) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in m["postprocess_rows"]:
                amp_label = f"{r['amplitude']}" + (" (intent rule on the clip's own audio arousal: " + ", ".join(
                    f"{k[:14]} {v:.2f}" for k, v in (r.get("audio_arousal") or {}).items()) + ")" if r.get("intent_from_audio") else "")
                for cond, v in r["conditions"].items():
                    per = "; ".join(f"{k[:14]}: {x['margin']:+.3f} / {x['stillness']:.3f}" for k, x in v["per_clip"].items())
                    L.append(f"| {r['settle_s']} | {r['pitch_floor']} | {amp_label} | {cond} | {v['beat_recall']:.3f} | {v['beat_recall_shuffled']:.3f} | "
                             f"{v['margin_min']:+.3f} | {v['stillness']:.3f} ({v['gt_stillness']:.3f}) | {v['w1_relative_mean']:.3f} | {per} |")
            L.append("")
            try:
                from .intent import EXAMPLE_LINES, LEXICON_VERSION, amplitude_for, analyse
                from .retrieval import AROUSAL_BONUS, THINKING_BONUS

                L.append(f"### Intent layer ({LEXICON_VERSION}: generic cue words + punctuation, no model)\n")
                L.append(f"Amplitude rule 0.8 + 0.5 * arousal (cap 1.3); retrieval arousal bonus {AROUSAL_BONUS}, thinking bonus {THINKING_BONUS}. "
                         "The lexicon stores none of the grader's utterances; the grader's lines below are read from "
                         "`animacy.grade.movements` at report time only.\n")
                L.append("| source | intended | line | tag | arousal | valence | amplitude |")
                L.append("|---|---|---|---|---|---|---|")
                rows = []
                try:
                    from ..grade.movements import MOVEMENTS  # type: ignore

                    rows += [("grader", mv.key, mv.text) for mv in MOVEMENTS]
                except Exception:  # noqa: BLE001
                    rows.append(("grader", "-", "(animacy.grade.movements not importable)"))
                rows += [("example", tag, s) for tag, lines in EXAMPLE_LINES.items() for s in lines]
                for src, intended, text in rows:
                    it = analyse(text)
                    L.append(f"| {src} | {intended} | {text} | {it.tag}{'' if it.tag == intended else ' (MISS)'} | {it.arousal:.2f} | {it.valence:+.2f} | {amplitude_for(it.arousal):.2f} |")
                L.append("")
            except Exception as e:  # noqa: BLE001
                L.append(f"(intent table unavailable: {e})\n")
        cols = [k for k in ("model", "ar", "retrieval") if k in ev["velocity"]]
        L.append("Per-channel W1 (deg/s or mm/s or unit/s):\n")
        L.append("| channel | gt mean speed | " + " | ".join(cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for i, ch in enumerate(MODEL_CHANNELS):
            row = [f"{ev['velocity']['gt_mean_speed'][i]:.3f}"] + [f"{ev['velocity'][k]['w1_per_channel'][i]:.3f}" for k in cols]
            L.append(f"| {ch} | " + " | ".join(row) + " |")
        L.append("")
        L.append("### Retarget legality (speed-cap violations after `retarget_clip`)\n")
        L.append("| robot / condition | violations | worst speed ratio | mean time stretch |")
        L.append("|---|---|---|---|")
        for k, v in ev["legality"].items():
            L.append(f"| {k} | {v['violations']} | {v['worst_speed_ratio']} | x{v['mean_time_stretch']} |")
        L.append("")
        vd = ev["verdict"]
        L.append("### Verdict\n")
        for arch, cd in vd.get("candidates", {}).items():
            L.append(f"- `{cd['condition']}` ({arch}): head-beat recall {_fmt(cd['beat_recall'])} vs shuffled {_fmt(cd['shuffled_beat_recall'])} "
                     f"(margin {_fmt(cd['margin'])}, required >= {vd['required_margin']}); NLL {_fmt(cd['nll'])} vs unigram floor {_fmt(cd['nll_unigram_floor'])}; "
                     f"stillness {_fmt(cd['stillness'])} (gt {ev['stillness']['gt']:.3f}); W1 rel {_fmt(cd['w1_relative_mean'])} -> "
                     + ("**qualifies**" if cd["qualifies"] else "**does not qualify** (" + ("shuffle margin" if not cd["beats_shuffle"] else "") + (" + " if not cd["beats_shuffle"] and not cd["below_floor"] else "") + ("NLL not below floor" if not cd["below_floor"] else "") + ")"))
        L.append(f"- retrieval head-beat recall {_fmt(vd['retrieval_beat_recall'])} vs shuffled {_fmt(vd['retrieval_shuffled_beat_recall'])}")
        promo = vd.get("promotion")
        if promo:
            L.append(f"- promotion rule: {promo['rule']}")
            for arch, cd in promo["candidates"].items():
                L.append(f"  - {arch}: " + "; ".join(f"{n}: margin {r['margin']:+.3f}, stillness {r['stillness']:.3f} vs gt {r['gt_stillness']:.3f}, "
                                                     f"NLL {_fmt(r['nll'])} vs floor {_fmt(r['floor'])} -> {'ok' if r['ok'] else 'fails'}"
                                                     for n, r in cd["per_clip"].items()) + f" => {'**QUALIFIES**' if cd['qualifies'] else 'does not qualify'}")
        L.append(f"- default backend for the demo: **{vd['default_backend']}**" + (f" (model arch for the selectable 'model' source: {vd['default_arch']})" if vd.get("default_arch") else ""))
        L.append("")
    ex = m.get("export")
    if ex:
        L.append("## Browser export\n")
        for k in ("a2m", "a2m_ar", "vq_decoder"):
            if k in ex:
                r = ex[k]
                if r.get("max_abs_diff") is None:
                    L.append(f"- `{os.path.basename(r['path'])}`: {r['bytes'] / 1e6:.2f} MB ({r.get('exporter', 'unchanged')})")
                    continue
                L.append(f"- `{os.path.basename(r['path'])}`: {r['bytes'] / 1e6:.2f} MB, exporter {r['exporter']}, onnxruntime vs torch max abs diff "
                         f"{r['max_abs_diff']:.2e} at L={r.get('verify_lengths')} -> {'OK' if r['ok'] else 'MISMATCH'}"
                         + (f" (fp16 weights; fp32 graph {r['max_abs_diff_fp32_graph']:.2e})" if r.get("fp16") else ""))
        if "retrieval" in ex:
            L.append(f"- retrieval index: {ex['retrieval']['n_windows']} windows, {ex['retrieval']['bytes'] / 1e6:.2f} MB")
        L.append(f"- bigram.bin {ex['bigram']['bytes'] / 1e6:.2f} MB; bundle total {ex['total_bytes'] / 1e6:.2f} MB -> `{os.path.dirname(ex['model_json'])}`")
        L.append("")
    L.append("## Timing\n")
    for k, v in m["timing_s"].items():
        L.append(f"- {k}: {v:.1f} s")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/clips")
    p.add_argument("--out", default="checkpoints/v1")
    p.add_argument("--cache-dir", default="checkpoints/feature_cache", help="audio feature cache ('' = off)")
    p.add_argument("--synthetic", action="store_true", help="fabricate clips under <out>/synthetic_clips and train on them")
    p.add_argument("--synthetic-clips", type=int, default=12)
    p.add_argument("--synthetic-seconds", type=float, default=20.0)
    p.add_argument("--arch", choices=["ff", "ar", "both"], default="both", help="which audio->codes predictors to train")
    # stage 1
    p.add_argument("--epochs-vq", type=int, default=40)
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--vq-dim", type=int, default=64)
    p.add_argument("--vq-revive-after", type=int, default=100, help="batches a code may idle before it is re-seeded")
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--init-vq", default="", help="warm-start (or, with --epochs-vq 0, reuse) a vq.pt; its stats are kept")
    # stage 2
    p.add_argument("--epochs-a2m", type=int, default=40)
    p.add_argument("--epochs-ar", type=int, default=40)
    p.add_argument("--patience", type=int, default=6, help="early stopping on held-out NLL (0 = off)")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4, help="audio trunk layers")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2, help="feed-forward model dropout")
    p.add_argument("--ar-layers", type=int, default=3, help="AR code-decoder layers")
    p.add_argument("--ar-dropout", type=float, default=0.25)
    p.add_argument("--ar-window", type=int, default=32, help="AR self-attention window (codes)")
    p.add_argument("--ar-chunk-frames", type=int, default=120, help="AR training chunk length (frames)")
    p.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES, help="feed-forward training chunk length (frames)")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="AR only")
    p.add_argument("--ar-change-weight", type=float, default=1.0, help="AR loss weight on steps whose code differs from the previous one (1 = off)")
    p.add_argument("--repeat-penalty", type=float, default=0.0, help="AR sampling: logit penalty on repeating the previous code")
    p.add_argument("--stay-bias", type=float, default=0.0, help="AR sampling: logit bonus on repeating the previous code while the audio is quiet")
    p.add_argument("--stay-energy", type=float, default=-0.3, help="quiet = normalised log energy below this")
    p.add_argument("--speaker-cap", type=float, default=0.4, help="max share of the training mass for one speaker (_index.json); 0 = off")
    p.add_argument("--no-fp16", action="store_true", help="export float32 weights (default: float16 initializers for the two predictors)")
    p.add_argument("--settle-s", type=float, default=0.5, help="utterance-final settle to neutral over the last N s of speech (0 = off)")
    p.add_argument("--pitch-floor", type=float, default=-3.0, help="low-frequency head_pitch mean never below this (deg); nan = off")
    p.add_argument("--amplitude", type=float, default=1.0, help="scale of the decoded motion (all channels)")
    p.add_argument("--no-ar-init-encoder", action="store_true", help="do not warm-start the AR audio trunk from the trained a2m")
    p.add_argument("--init-a2m", default="", help="warm-start a2m.pt")
    p.add_argument("--init-ar", default="", help="warm-start a2m_ar.pt")
    p.add_argument("--feat-noise", type=float, default=0.2)
    p.add_argument("--time-mask", type=int, default=2, help="max blanked spans per chunk (0 = off)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--chunk-stride", type=int, default=15, help="frames between training chunks")
    p.add_argument("--no-mirror", action="store_true", help="disable left-right mirror augmentation")
    p.add_argument("--warp", type=float, default=0.08, help="time-warp augmentation +-fraction (0 = off)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--a2m-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="VQ learning rate")
    p.add_argument("--a2m-lr", type=float, default=2e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--holdout", nargs="*", default=[], help="clip or subject names to hold out (overrides --val-frac)")
    p.add_argument("--exclude", nargs="*", default=[], help="clip names to drop entirely")
    p.add_argument("--max-retrieval", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bigram-weight", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-eval-runs", type=int, default=0, help="0 = all held-out runs")
    p.add_argument("--sweep", action="store_true", help="also evaluate a sampling grid on the held-out split")
    p.add_argument("--export-dir", default="web/models")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--robots", nargs="*", default=["lamp", "reachy_mini"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.pitch_floor is not None and args.pitch_floor != args.pitch_floor:      # nan = off
        args.pitch_floor = None
    command = "python -m animacy.model.train " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")
    log = lambda s: _log(logf, s)  # noqa: E731
    log(f"== animacy.model.train  device={device}  torch {torch.__version__}  out={args.out}")
    log(f"   {command}")
    timing: Dict[str, float] = {}
    flags: List[str] = []
    t0 = time.time()
    do_ff = args.arch in ("ff", "both")
    do_ar = args.arch in ("ar", "both")

    # --- data
    if args.synthetic:
        clip_dir = os.path.join(args.out, "synthetic_clips")
        make_synthetic_clips(clip_dir, n_clips=args.synthetic_clips, seconds=args.synthetic_seconds, seed=args.seed)
        log(f"-- synthetic clips written to {clip_dir}")
    else:
        clip_dir = args.data
    log(f"-- loading clips from {clip_dir}")
    clips = load_clips(clip_dir, cache_dir=args.cache_dir or None)
    if args.exclude:
        dropped = [c.name for c in clips if c.name in set(args.exclude)]
        clips = [c for c in clips if c.name not in set(args.exclude)]
        log(f"   excluded {dropped} (--exclude)")
    if not clips:
        log("no usable clips (need motion.parquet + audio.wav with face_valid runs >= 1 s). Try --synthetic.")
        return 2
    summary = summarise(clips)
    is_real = any(c.source != "synthetic" for c in clips)
    log(f"   {summary['n_clips']} clips, {summary['valid_minutes']} valid minutes ({summary['valid_minutes_with_audio']} with audio), "
        f"sources {summary['sources']}")
    if summary["valid_minutes_with_audio"] < 3.0 and is_real:
        flags.append(f"TINY_DATA: only {summary['valid_minutes_with_audio']} valid minutes with audio; numbers below are not representative")
    train_clips, val_clips, split_info = split_clips(clips, args.val_frac, args.seed, holdout=args.holdout or None)
    if split_info.get("leaky"):
        flags.append("LEAKY_SPLIT: a single subject/clip was split in time; held-out numbers overstate generalisation")
    if not val_clips:
        flags.append("NO_HELDOUT_SPLIT: too little data to hold anything out; metrics are computed on the training clips")
        val_clips = train_clips
    log(f"   split by {split_info['mode']}: train {len(train_clips)} / val {len(val_clips)} clips; held out {split_info['held_out_groups']}")
    n_train_clips = len(train_clips)
    augmentation = {"mirror": not args.no_mirror, "warp": args.warp, "feat_noise": args.feat_noise, "time_mask": args.time_mask}
    cap_info = None
    if args.speaker_cap > 0 and not args.synthetic:
        speakers = load_speaker_index(args.data)
        train_clips, cap_info = apply_speaker_cap(train_clips, speakers, args.speaker_cap)
        log(f"   speaker cap {args.speaker_cap:.0%} ({len(speakers)} clips in _index.json): minutes before {cap_info['minutes_before']}")
        log(f"     effective {cap_info['minutes_effective']}; keep-probabilities {cap_info['weights'] or 'none needed'}")
        augmentation["speaker_cap"] = cap_info
    if not args.no_mirror:
        train_clips = train_clips + [mirror_clip(c) for c in train_clips]
        log(f"   mirror augmentation: {n_train_clips} -> {len(train_clips)} training clips")
    warps = (1.0, 1.0 - args.warp, 1.0 + args.warp) if args.warp > 0 else (1.0,)
    init_vq = None
    if args.init_vq:
        init_vq = MotionVQVAE.load(args.init_vq, device)
        stats = init_vq.stats
        log(f"   VQ warm start from {args.init_vq}: keeping its standardisation stats")
    else:
        stats = compute_stats(train_clips)
    timing["data"] = time.time() - t0

    # --- stage 1
    t1 = time.time()
    log("-- stage 1: VQ tokenizer")
    segs_tr, segs_va = vq_segments(train_clips, stats, rng=np.random.default_rng(args.seed)), vq_segments(val_clips, stats)
    if init_vq is not None and args.epochs_vq == 0:
        vq, vq_hist, vq_base = init_vq.eval(), [], (float(np.abs(segs_va).mean()) if len(segs_va) else float("nan"))
        log(f"  reusing {args.init_vq} unchanged (--epochs-vq 0)")
    else:
        vq, vq_hist, vq_base = train_vq(segs_tr, segs_va, stats, args, device, log, init=init_vq)
    seqs_tr = run_code_sequences(train_clips, stats, vq.encode)
    seqs_va = run_code_sequences(val_clips, stats, vq.encode)
    used_tr, ppl_tr = vq.quantizer.usage(torch.as_tensor(np.concatenate(seqs_tr)))
    used_va, ppl_va = vq.quantizer.usage(torch.as_tensor(np.concatenate(seqs_va))) if seqs_va else (0, 0.0)
    log(f"   codes used on whole training runs: {used_tr}/{vq.n_codes} (perplexity {ppl_tr:.1f}); held-out {used_va} ({ppl_va:.1f})")
    if is_real and used_tr < MIN_USED_CODES_REAL:
        flags.append(f"VQ_CODEBOOK_UNDERUSED: {used_tr} < {MIN_USED_CODES_REAL} codes used on real data")
    vq_info = {"n_codes": vq.n_codes, "dim": vq.config["dim"], "n_train_segments": int(len(segs_tr)), "n_val_segments": int(len(segs_va)),
               "best_val_mae": min((h.get("val_mae", float("inf")) for h in vq_hist), default=None), "baseline_val_mae": vq_base,
               "used_codes_train": used_tr, "perplexity_train": ppl_tr, "used_codes_val": used_va, "perplexity_val": ppl_va,
               "init": args.init_vq or None, "history": vq_hist}
    vq.save(os.path.join(args.out, "vq.pt"), {"stats": {k: v.tolist() for k, v in stats.items()}, "info": {k: v for k, v in vq_info.items() if k != "history"}})
    timing["vq"] = time.time() - t1

    # --- stage 2
    t2 = time.time()
    bigram = BigramPrior(vq.n_codes).fit(seqs_tr)
    bigram_logp, unigram = bigram.log_probs(), bigram.unigram()
    a2m, a2m_info, ar, ar_info = None, None, None, None
    training_info = {"command": command, "device": device, "torch": torch.__version__, "data": summary, "split": split_info,
                     "augmentation": augmentation, "vq": {k: v for k, v in vq_info.items() if k != "history"}}
    if do_ff:
        log("-- stage 2a: audio -> codes (feed-forward a2m)")
        ch_tr = a2m_chunks(train_clips, stats, vq.encode, chunk=args.chunk_frames, stride=args.chunk_stride, warps=warps,
                           rng=np.random.default_rng(args.seed + 1))
        ch_va = a2m_chunks(val_clips, stats, vq.encode, chunk=args.chunk_frames, stride=args.chunk_frames // 2)
        if len(ch_tr["codes"]) == 0:
            log("no audio-aligned chunks to train on (do the clips have audio.wav?)")
            return 2
        if args.init_a2m:
            a2m, _ = AudioToMotion.load(args.init_a2m, device)
            a2m = a2m.to(device)
        else:
            a2m = AudioToMotion(n_codes=vq.n_codes, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout).to(device)
        log(f"  a2m {sum(p.numel() for p in a2m.parameters()) / 1e6:.2f}M params; train {len(ch_tr['codes'])} / val {len(ch_va['codes'])} chunks of "
            f"{args.chunk_frames} frames ({int(ch_tr['mask'].sum())} / {int(ch_va['mask'].sum())} code steps)" + (f"; warm start {args.init_a2m}" if args.init_a2m else ""))
        hist = _fit(a2m, "ff", ch_tr, ch_va, args, device, log, args.epochs_a2m, args.a2m_lr, args.patience)
        a2m_info = {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads, "chunk_frames": args.chunk_frames,
                    "n_train_chunks": int(len(ch_tr["codes"])), "n_val_chunks": int(len(ch_va["codes"])), "init": args.init_a2m or None, "history": hist}
        training_info["a2m"] = {k: v for k, v in a2m_info.items() if k != "history"}
        a2m.save(os.path.join(args.out, "a2m.pt"), {"bigram_logp": torch.from_numpy(bigram_logp),
                                                     "bigram_counts": torch.from_numpy(bigram.counts.astype(np.float32)),
                                                     "unigram": torch.from_numpy(unigram), "info": training_info})
    timing["a2m"] = time.time() - t2
    t3 = time.time()
    if do_ar:
        log("-- stage 2b: audio -> codes (autoregressive a2m_ar)")
        ch_tr = a2m_chunks(train_clips, stats, vq.encode, chunk=args.ar_chunk_frames, stride=args.chunk_stride, warps=warps,
                           rng=np.random.default_rng(args.seed + 2))
        ch_va = a2m_chunks(val_clips, stats, vq.encode, chunk=args.ar_chunk_frames, stride=args.ar_chunk_frames // 2)
        if len(ch_tr["codes"]) == 0:
            log("no audio-aligned chunks to train on (do the clips have audio.wav?)")
            return 2
        if args.init_ar:
            ar, _ = AudioToMotionAR.load(args.init_ar, device)
            ar = ar.to(device)
        else:
            ar = AudioToMotionAR(n_codes=vq.n_codes, d_model=args.d_model, enc_layers=args.n_layers, dec_layers=args.ar_layers,
                                 n_heads=args.n_heads, dropout=args.ar_dropout, window=args.ar_window).to(device)
            enc_src = a2m
            if enc_src is None and args.init_a2m:
                enc_src = AudioToMotion.load(args.init_a2m, device)[0]
            if enc_src is not None and not args.no_ar_init_encoder and enc_src.config["d_model"] == args.d_model and enc_src.config["n_layers"] == args.n_layers:
                ar.encoder.load_state_dict(enc_src.state_dict())
                log(f"  AR audio trunk warm-started from {'the trained a2m' if a2m is not None else args.init_a2m}")
        log(f"  a2m_ar {sum(p.numel() for p in ar.parameters()) / 1e6:.2f}M params; train {len(ch_tr['codes'])} / val {len(ch_va['codes'])} chunks of "
            f"{args.ar_chunk_frames} frames ({int(ch_tr['mask'].sum())} / {int(ch_va['mask'].sum())} code steps)" + (f"; warm start {args.init_ar}" if args.init_ar else ""))
        hist = _fit(ar, "ar", ch_tr, ch_va, args, device, log, args.epochs_ar, args.a2m_lr, args.patience)
        ar_info = {"d_model": args.d_model, "enc_layers": args.n_layers, "dec_layers": args.ar_layers, "n_heads": args.n_heads,
                   "dropout": args.ar_dropout, "window": args.ar_window, "label_smoothing": args.label_smoothing, "patience": args.patience,
                   "chunk_frames": args.ar_chunk_frames, "n_train_chunks": int(len(ch_tr["codes"])), "n_val_chunks": int(len(ch_va["codes"])),
                   "init": args.init_ar or None, "history": hist}
        training_info["ar"] = {k: v for k, v in ar_info.items() if k != "history"}
        ar.save(os.path.join(args.out, "a2m_ar.pt"), {"unigram": torch.from_numpy(unigram), "info": training_info})
    timing["ar"] = time.time() - t3
    info = {"channels": list(MODEL_CHANNELS), "stats": {k: v.tolist() for k, v in stats.items()}, "unigram": unigram.tolist(),
            "archs": [a for a, m in (("ff", a2m), ("ar", ar)) if m is not None], "default_arch": "ar" if ar is not None else "ff",
            "training": training_info}
    with open(os.path.join(args.out, "model_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)

    # --- retrieval baseline
    t4 = time.time()
    index = RetrievalIndex.build(train_clips, max_windows=args.max_retrieval, seed=args.seed)
    index.save(args.out, "retrieval")
    log(f"-- retrieval index: {len(index)} windows (from {index.meta.get('n_source_windows', len(index))})")
    timing["retrieval"] = time.time() - t4

    # --- metrics
    t5 = time.time()
    log("-- held-out metrics")
    model = MotionModel.load(args.out, device)
    profiles = []
    for r in args.robots:
        p = r if os.path.exists(r) else os.path.join(robots_root(), r)
        try:
            profiles.append(load_profile(p))
        except Exception as e:  # noqa: BLE001
            log(f"   could not load robot profile {r}: {e}")
    # the headline metrics are the raw generators (no settle / floor / amplitude); those are separate rows
    ev = evaluate(model, index, val_clips, profiles, seed=args.seed, temperature=args.temperature,
                  bigram_weight=args.bigram_weight, top_p=args.top_p, repeat_penalty=args.repeat_penalty,
                  stay_bias=args.stay_bias, stay_energy=args.stay_energy, max_runs=args.max_eval_runs or None)
    for arch, cd in ev["verdict"]["candidates"].items():
        if not cd["qualifies"]:
            flags.append(f"{ARCH_COND[arch].upper()}_DOES_NOT_QUALIFY (pooled): shuffle margin {cd['margin']:+.3f} (need >= {ev['verdict']['required_margin']}), "
                         f"NLL {cd['nll']:.3f} vs floor {cd['nll_unigram_floor']:.3f}")
    sampling = {"temperature": args.temperature, "bigram_weight": args.bigram_weight, "top_p": args.top_p,
                "repeat_penalty": args.repeat_penalty, "stay_bias": args.stay_bias, "stay_energy": args.stay_energy}
    # --- per held-out clip (speaker): the promotion rule needs every speaker, not the pool
    per_clip: Dict[str, Dict] = {}
    if len(val_clips) > 1:
        log("-- per held-out clip")
        for c in val_clips:
            e_c = evaluate(model, index, [c], [], seed=args.seed, temperature=args.temperature, bigram_weight=args.bigram_weight,
                           top_p=args.top_p, repeat_penalty=args.repeat_penalty, stay_bias=args.stay_bias,
                           stay_energy=args.stay_energy, verbose=False)
            per_clip[c.name] = compact(e_c)
            pc = per_clip[c.name]
            for cond in ("model", "ar", "retrieval"):
                if cond in pc["conditions"]:
                    x, xs = pc["conditions"][cond], pc["conditions"][f"{cond}_shuffled"]
                    log(f"   {c.name:28s} {cond:9s} beat {x['beat_recall']:.3f} vs shuffled {xs['beat_recall']:.3f} "
                        f"({x['beat_recall'] - xs['beat_recall']:+.3f})  still {x['stillness']:.3f} (gt {pc['gt_stillness']:.3f})  "
                        f"W1 rel {x['w1_relative_mean']:.3f}" + (f"  NLL {x['nll']:.3f} vs floor {pc['nll_unigram_floor']:.3f}" if x["nll"] is not None else ""))
    else:
        per_clip[val_clips[0].name] = compact(ev)
    ev["per_clip"] = per_clip
    promo = promotion_verdict(per_clip)
    ev["verdict"]["promotion"] = promo
    ev["verdict"]["default_backend"] = promo["default_backend"]
    log(f"   promotion rule ({promo['rule']}): " + "; ".join(
        f"{a}: {'QUALIFIES' if c['qualifies'] else 'no'} " + str({n: round(r['margin'], 3) for n, r in c['per_clip'].items()})
        for a, c in promo["candidates"].items()) + f" -> default backend = {promo['default_backend']}")
    if promo["default_backend"] == "retrieval":
        flags.append("RETRIEVAL_IS_DEFAULT: no learned generator met the promotion rule on every held-out speaker")
    else:
        flags.append(f"PROMOTED: {promo['default_backend']} met the promotion rule on every held-out speaker")
    info["default_arch"] = ev["verdict"].get("default_arch") or info["default_arch"]

    sweep = []
    if args.sweep:
        log("-- sampling sweep, per held-out clip (learned generators only; the defaults above stay the reported numbers)")
        grid = []
        if a2m is not None:
            grid += [("ff", T, w, 0.0, 0.0) for T in (0.5, 0.8, 1.0) for w in (0.0, 0.5, 1.0)]
        if ar is not None:
            grid += [("ar", T, tp, args.repeat_penalty, args.stay_bias) for T in (0.8, 1.0) for tp in (0.9, 1.0)]
            grid += [("ar", 1.0, 1.0, rp, args.stay_bias) for rp in (0.5, 1.0, 2.0) if rp != args.repeat_penalty]
            # the stillness knob: reported as its own rows, never folded into the defaults here
            grid += [("ar", args.temperature, args.top_p, args.repeat_penalty, sb) for sb in (1.0, 2.0, 3.0) if sb != args.stay_bias]
        for arch, T, second, rp, sb in grid:
            kw = dict(temperature=T, bigram_weight=second if arch == "ff" else args.bigram_weight,
                      top_p=second if arch == "ar" else args.top_p, repeat_penalty=rp, stay_bias=sb, stay_energy=args.stay_energy)
            cn = ARCH_COND[arch]
            rows_c = {}
            for c in val_clips:
                e2 = evaluate(model, None, [c], [], seed=args.seed, archs=[arch], max_runs=args.max_eval_runs or None, verbose=False, **kw)
                b, bA = e2["beat"], e2["beat_all_channels"]
                rows_c[c.name] = {"head_beat_recall": b[cn]["recall"], "head_beat_recall_shuffled": b[f"{cn}_shuffled"]["recall"],
                                  "head_beat_precision": b[cn]["precision"], "all_beat_recall": bA[cn]["recall"],
                                  "all_beat_recall_shuffled": bA[f"{cn}_shuffled"]["recall"], "stillness": e2["stillness"][cn],
                                  "gt_stillness": e2["stillness"]["gt"], "w1_relative_mean": e2["velocity"][cn]["w1_relative_mean"]}
            n = len(rows_c)
            mean = {k: sum(r[k] for r in rows_c.values()) / n for k in next(iter(rows_c.values()))}
            row = {"arch": arch, "temperature": T, "second": second if arch == "ff" else f"{second} / rp {rp} / stay {sb}",
                   "top_p": second if arch == "ar" else None, "repeat_penalty": rp, "stay_bias": sb,
                   "per_clip": rows_c, "mean_margin": mean["head_beat_recall"] - mean["head_beat_recall_shuffled"],
                   "min_margin": min(r["head_beat_recall"] - r["head_beat_recall_shuffled"] for r in rows_c.values()),
                   "max_still_gap": max(abs(r["stillness"] - r["gt_stillness"]) for r in rows_c.values()), **mean}
            sweep.append(row)
            log(f"   {arch} T={T} {'w' if arch == 'ff' else 'top_p'}={second}{'' if arch == 'ff' else f' rp={rp} stay={sb}'}: mean head-beat recall "
                f"{row['head_beat_recall']:.3f} vs shuffled {row['head_beat_recall_shuffled']:.3f} (mean margin {row['mean_margin']:+.3f}, min {row['min_margin']:+.3f}); "
                f"still {row['stillness']:.3f} (max gap {row['max_still_gap']:.3f}); W1 rel {row['w1_relative_mean']:.3f}  "
                + "  ".join(f"{k[:20]} {v['head_beat_recall'] - v['head_beat_recall_shuffled']:+.3f}/still {v['stillness']:.3f}" for k, v in rows_c.items()))
        # the shipped AR top-p: 0.9 vs 1.0 at T=1.0, picked by the mean margin over the held-out speakers
        cand = [r for r in sweep if r["arch"] == "ar" and r["temperature"] == 1.0 and r["repeat_penalty"] == args.repeat_penalty
                and r["stay_bias"] == args.stay_bias and r["top_p"] in (0.9, 1.0)]
        if cand:
            best = max(cand, key=lambda r: r["mean_margin"])
            sampling["top_p"] = best["top_p"]
            sampling["picked_by"] = f"mean head-beat margin over {len(val_clips)} held-out clips: " + ", ".join(
                f"top_p {r['top_p']} -> {r['mean_margin']:+.3f}" for r in cand)
            log(f"   shipped AR sampling: T=1.0 top_p={best['top_p']} rp={args.repeat_penalty} ({sampling['picked_by']})")
    info["sampling"] = sampling
    info["postprocess"] = {"settle_s": args.settle_s, "pitch_floor": args.pitch_floor, "amplitude": args.amplitude}
    with open(os.path.join(args.out, "model_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    pp_rows = []
    if args.sweep:
        log("-- generation-side options (settle / pitch floor / amplitude), per held-out clip, at the shipped sampling")
        pp_rows = postprocess_rows(model, index, val_clips, sampling, seed=args.seed)
        for r in pp_rows:
            log(f"   settle {r['settle_s']}s pitch_floor {r['pitch_floor']} amp {r['amplitude']}{' intent-from-audio' if r.get('intent_from_audio') else ''}: " + "; ".join(
                f"{cond} beat {v['beat_recall']:.3f} vs {v['beat_recall_shuffled']:.3f} (min margin {v['margin_min']:+.3f}) still {v['stillness']:.3f} "
                f"(gt {v['gt_stillness']:.3f}) W1 {v['w1_relative_mean']:.3f}" for cond, v in r["conditions"].items()))
    timing["metrics"] = time.time() - t5

    metrics = {"command": command, "device": device, "torch": torch.__version__, "flags": flags, "data": summary, "split": split_info,
               "excluded": list(args.exclude), "augmentation": augmentation, "sampling": sampling,
               "n_train_clips": n_train_clips, "n_val_clips": len(val_clips), "vq": vq_info, "a2m": a2m_info, "ar": ar_info,
               "eval": ev, "sampling_sweep": sweep, "postprocess_rows": pp_rows, "postprocess": info["postprocess"], "timing_s": timing}

    # --- export
    if not args.no_export:
        t6 = time.time()
        log(f"-- export -> {args.export_dir}")
        try:
            from .export import export_bundle

            model.info = info
            rep = export_bundle(model, index, args.export_dir, ev, fp16=not args.no_fp16)
            for k in ("a2m", "a2m_ar", "vq_decoder"):
                if k in rep:
                    r = rep[k]
                    extra = ""
                    if r.get("fp16"):
                        extra = f" [fp16 weights {r['bytes_fp32'] / 1e6:.2f} -> {r['bytes'] / 1e6:.2f} MB; fp32-graph diff {r['max_abs_diff_fp32_graph']:.2e}" \
                                + (f"; code agreement {r['code_agreement_fp16']}" if "code_agreement_fp16" in r else "") + "]"
                    log(f"   {os.path.basename(r['path'])}: {r['bytes'] / 1e6:.2f} MB ({r['exporter']}), max abs diff vs torch {r['max_abs_diff']:.2e} "
                        f"(tol {r['tol']}) -> {'OK' if r['ok'] else 'MISMATCH'}{extra}")
                    if not r["ok"]:
                        flags.append(f"ONNX_MISMATCH: {k} max abs diff {r['max_abs_diff']:.2e} > {r['tol']}" + (" (fp16 weights)" if r.get("fp16") else ""))
            if "retrieval" in rep:
                log(f"   retrieval index {rep['retrieval']['n_windows']} windows, {rep['retrieval']['bytes'] / 1e6:.2f} MB; bundle {rep['total_bytes'] / 1e6:.2f} MB")
            metrics["export"] = rep
        except ImportError as e:
            flags.append(f"EXPORT_SKIPPED: {e}; run `python -m animacy.model.export --ckpt {args.out} --out {args.export_dir}` in a venv with onnx + onnxruntime")
            log(f"   export skipped: {e}")
        timing["export"] = time.time() - t6
    metrics["timing_s"] = timing
    metrics["flags"] = flags

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1, default=float)
    write_report(os.path.join(args.out, "REPORT.md"), metrics, command)
    log(f"-- wrote {os.path.join(args.out, 'metrics.json')} and REPORT.md  (total {time.time() - t0:.1f} s)")
    for f in flags:
        log(f"   FLAG: {f}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
