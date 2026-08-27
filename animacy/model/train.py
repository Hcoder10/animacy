"""Train the motion model: stage 1 VQ tokenizer, stage 2 audio -> codes, then the
held-out metrics from ``docs/MODEL.md``, then the browser export.

    python -m animacy.model.train --data data/clips --out checkpoints/v1
    python -m animacy.model.train --synthetic --out checkpoints/synthetic --epochs-vq 3 --epochs-a2m 3

Writes ``<out>/vq.pt``, ``<out>/a2m.pt``, ``<out>/model_info.json``,
``<out>/metrics.json``, ``<out>/REPORT.md`` and (unless ``--no-export``) the
``web/models/`` bundle. Nothing is claimed as measured that is not in metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..profile import load_profile, robots_root
from ..schema import RATE_HZ
from .a2m import AudioToMotion, BigramPrior
from .data import (CHUNK_FRAMES, MODEL_CHANNELS, SEGMENT, a2m_chunks, compute_stats, load_clips,
                   make_synthetic_clips, run_code_sequences, split_clips, summarise, vq_segments)
from .export import export_bundle
from .infer import MotionModel
from .metrics import evaluate
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
def train_vq(segs_tr: np.ndarray, segs_va: np.ndarray, stats, args, device: str, log) -> tuple:
    model = MotionVQVAE(n_codes=args.n_codes, dim=args.vq_dim, revive_after=args.vq_revive_after).to(device)
    model.set_stats(stats)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(segs_tr)
    xv = torch.from_numpy(segs_va).to(device) if len(segs_va) else None
    base = float(np.abs(segs_va).mean()) if len(segs_va) else float("nan")   # predict the (zero) mean
    log(f"  VQ-VAE {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {args.n_codes} codes x {args.vq_dim}d; "
        f"train {len(segs_tr)} / val {len(segs_va)} segments of {SEGMENT} frames; baseline val MAE {base:.4f}")
    history: List[Dict] = []
    best, best_state = float("inf"), None
    g = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs_vq + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(xt), args.batch_size):
            xb = xt[perm[i:i + args.batch_size]].to(device)
            recon, commit, _ = model(xb)
            loss = F.mse_loss(recon, xb) + args.commit_weight * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(xb)
            nb += 1
        model.eval()
        rec: Dict = {"epoch": epoch, "train_loss": tot / max(len(xt), 1)}
        with torch.no_grad():
            if xv is not None:
                recon, _, idx = model(xv)
                rec["val_mae"] = float((recon - xv).abs().mean())
                rec["val_used_codes"], rec["val_perplexity"] = model.quantizer.usage(idx)
            # usage on the training set, the number that must not collapse
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
            f"codes used (train) {rec['train_used_codes']}/{args.n_codes}  perplexity {rec['train_perplexity']:.1f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, base


# ---------------------------------------------------------------------------
# stage 2
# ---------------------------------------------------------------------------
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


def train_a2m(chunks_tr: Dict[str, np.ndarray], chunks_va: Dict[str, np.ndarray], args, device: str, log) -> tuple:
    model = AudioToMotion(n_codes=args.n_codes, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                          dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    tr = {k: torch.from_numpy(v) for k, v in chunks_tr.items()}
    va = {k: torch.from_numpy(v).to(device) for k, v in chunks_va.items()}
    n_tr, n_va = len(tr["codes"]), len(va["codes"])
    log(f"  a2m {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params; train {n_tr} / val {n_va} chunks of "
        f"{CHUNK_FRAMES} frames ({int(chunks_tr['mask'].sum())} / {int(chunks_va['mask'].sum())} code steps)")
    rng = np.random.default_rng(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    history: List[Dict] = []
    best, best_state = float("inf"), None
    bs = args.a2m_batch_size
    for epoch in range(1, args.epochs_a2m + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=g)
        tot, nb, n_causal = 0.0, 0, 0
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            f = tr["features"][idx].to(device)
            if args.feat_noise > 0:
                f = f + args.feat_noise * torch.randn_like(f)
            s, c, m = tr["speaking"][idx].to(device), tr["codes"][idx].to(device), tr["mask"][idx].to(device)
            causal = bool(rng.random() < 0.5)          # one set of weights serves talk and listen
            n_causal += int(causal)
            lg = model(f, s, causal, key_padding_mask=~m)
            loss = F.cross_entropy(lg[m], c[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        rec: Dict = {"epoch": epoch, "train_loss": tot / max(nb, 1), "causal_batches": n_causal, "batches": nb}
        if n_va:
            rec["val_noncausal"] = _eval_a2m(model, va, False)
            rec["val_causal"] = _eval_a2m(model, va, True)
            score = 0.5 * (rec["val_noncausal"]["nll"] + rec["val_causal"]["nll"])
        else:
            score = rec["train_loss"]
        history.append(rec)
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        vn, vc = rec.get("val_noncausal", {}), rec.get("val_causal", {})
        log(f"  a2m epoch {epoch:3d} train CE {rec['train_loss']:.4f}  val NLL non-causal {vn.get('nll', float('nan')):.4f} "
            f"(top1 {vn.get('acc', float('nan')):.3f})  causal {vc.get('nll', float('nan')):.4f} (top1 {vc.get('acc', float('nan')):.3f})")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(path: str, m: Dict, command: str) -> None:
    d, sp, vq, a2m = m["data"], m["split"], m["vq"], m["a2m"]
    L = []
    L.append("# animacy motion model v1 - training report\n")
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
    L.append(f"- split: by {sp['mode']}; held out {sp['held_out_groups']} -> train {m['n_train_clips']} / val {m['n_val_clips']} clips"
             + ("  **(LEAKY: time split inside a single group)**" if sp.get("leaky") else ""))
    L.append("")
    L.append("## Stage 1: VQ tokenizer\n")
    L.append(f"- {vq['n_codes']} codes x {vq['dim']}d, one code per 2 frames, {vq['n_train_segments']} train / {vq['n_val_segments']} val segments of 8 frames")
    L.append(f"- best val MAE (standardised units) {_fmt(vq['best_val_mae'], 4)} vs predict-mean baseline {_fmt(vq['baseline_val_mae'], 4)}")
    L.append(f"- codes used on whole training runs: **{vq['used_codes_train']}/{vq['n_codes']}** (perplexity {_fmt(vq['perplexity_train'], 1)}); "
             f"on held-out runs {vq['used_codes_val']} (perplexity {_fmt(vq['perplexity_val'], 1)})")
    L.append("")
    L.append("| epoch | train loss | val MAE | used (train) | perplexity |")
    L.append("|---|---|---|---|---|")
    for r in vq["history"]:
        L.append(f"| {r['epoch']} | {r['train_loss']:.5f} | {_fmt(r.get('val_mae'), 4)} | {r['train_used_codes']} | {r['train_perplexity']:.1f} |")
    L.append("")
    L.append("## Stage 2: audio -> codes\n")
    L.append(f"- Transformer d {a2m['d_model']}, {a2m['n_layers']} layers, {a2m['n_heads']} heads; {a2m['n_train_chunks']} train / {a2m['n_val_chunks']} val chunks (2 s); causal/non-causal 50/50")
    L.append("")
    L.append("| epoch | train CE | val NLL (talk) | top1 | val NLL (listen) | top1 |")
    L.append("|---|---|---|---|---|---|")
    for r in a2m["history"]:
        vn, vc = r.get("val_noncausal", {}), r.get("val_causal", {})
        L.append(f"| {r['epoch']} | {r['train_loss']:.4f} | {_fmt(vn.get('nll'))} | {_fmt(vn.get('acc'))} | {_fmt(vc.get('nll'))} | {_fmt(vc.get('acc'))} |")
    L.append("")
    ev = m.get("eval", {})
    if "codes" in ev:
        c, ho = ev["codes"], ev["held_out"]
        L.append("## Held-out metrics\n")
        L.append(f"Held out: {ho['n_clips']} clips, {ho['n_runs']} runs, {ho['seconds']} s, {ho['code_steps']} code steps.\n")
        L.append("### Codes (lower NLL / higher top-1 is better)\n")
        L.append("| predictor | NLL | top-1 |")
        L.append("|---|---|---|")
        L.append(f"| unigram / majority floor | {_fmt(c['nll_unigram_floor'])} | {_fmt(c['top1_majority_floor'])} |")
        L.append(f"| retrieval (eps-smoothed 0.05) | {_fmt(c['nll_retrieval_eps0.05'])} | {_fmt(c['top1_retrieval'])} |")
        L.append(f"| model, talk (non-causal) | {_fmt(c['nll_model'])} | {_fmt(c['top1_model'])} |")
        L.append(f"| model, listen (causal) | {_fmt(c['nll_model_causal'])} | {_fmt(c['top1_model_causal'])} |")
        L.append("")
        L.append("### Motion statistics vs ground truth\n")
        L.append(f"Beat = head |angular velocity| peaks (prominence {ev['beat']['prominence_deg_per_s']:.1f} deg/s) within +-150 ms. "
                 f"Stillness = fraction of frames with head speed < {ev['stillness']['threshold_deg_per_s']} deg/s (ground truth {ev['stillness']['gt']:.3f}). "
                 "W1 = mean over channels of Wasserstein-1 between |velocity| histograms, relative to the ground-truth mean speed.\n")
        L.append("| condition | beat recall | beat precision | gt/gen peaks | stillness | W1 rel |")
        L.append("|---|---|---|---|---|---|")
        for cond in ("model", "model_shuffled", "model_causal", "retrieval", "retrieval_shuffled"):
            if cond in ev["beat"]:
                b, v = ev["beat"][cond], ev["velocity"][cond]
                L.append(f"| {cond} | {b['recall']:.3f} | {b['precision']:.3f} | {b['n_gt_peaks']}/{b['n_gen_peaks']} | {ev['stillness'][cond]:.3f} | {v['w1_relative_mean']:.3f} |")
        L.append("")
        L.append("Per-channel W1 (deg/s or mm/s or unit/s):\n")
        L.append("| channel | gt mean speed | " + " | ".join(k for k in ("model", "model_shuffled", "retrieval") if k in ev["velocity"]) + " |")
        L.append("|---|---|" + "---|" * len([k for k in ("model", "model_shuffled", "retrieval") if k in ev["velocity"]]))
        for i, ch in enumerate(MODEL_CHANNELS):
            row = [f"{ev['velocity']['gt_mean_speed'][i]:.3f}"]
            for k in ("model", "model_shuffled", "retrieval"):
                if k in ev["velocity"]:
                    row.append(f"{ev['velocity'][k]['w1_per_channel'][i]:.3f}")
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
        L.append(f"- model beat recall {_fmt(vd['model_beat_recall'])} vs shuffled-audio {_fmt(vd['model_shuffled_beat_recall'])} -> "
                 + ("**model beats its shuffle**" if vd["model_beats_shuffled_audio_on_beat_recall"] else "**MODEL DOES NOT BEAT SHUFFLED AUDIO**"))
        L.append(f"- retrieval beat recall {_fmt(vd['retrieval_beat_recall'])} vs shuffled {_fmt(vd['retrieval_shuffled_beat_recall'])}")
        L.append(f"- default backend for the demo: **{vd['default_backend']}**")
        L.append("")
    ex = m.get("export")
    if ex:
        L.append("## Browser export\n")
        for k in ("a2m", "vq_decoder"):
            r = ex[k]
            L.append(f"- `{os.path.basename(r['path'])}`: {r['bytes'] / 1e6:.2f} MB, exporter {r['exporter']}, onnxruntime vs torch max abs diff "
                     f"{r['max_abs_diff']:.2e} at L={r['verify_lengths']} -> {'OK' if r['ok'] else 'MISMATCH'}")
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
    p.add_argument("--synthetic", action="store_true", help="fabricate clips under <out>/synthetic_clips and train on them")
    p.add_argument("--synthetic-clips", type=int, default=12)
    p.add_argument("--synthetic-seconds", type=float, default=20.0)
    p.add_argument("--epochs-vq", type=int, default=40)
    p.add_argument("--epochs-a2m", type=int, default=40)
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--vq-dim", type=int, default=64)
    p.add_argument("--vq-revive-after", type=int, default=100, help="batches a code may idle before it is re-seeded")
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feat-noise", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--a2m-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-retrieval", type=int, default=3000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--bigram-weight", type=float, default=0.5)
    p.add_argument("--max-eval-runs", type=int, default=0, help="0 = all held-out runs")
    p.add_argument("--export-dir", default="web/models")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--robots", nargs="*", default=["lamp", "reachy_mini"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    command = "python -m animacy.model.train " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")
    log = lambda s: _log(logf, s)  # noqa: E731
    log(f"== animacy.model.train  device={device}  out={args.out}")
    log(f"   {command}")
    timing: Dict[str, float] = {}
    flags: List[str] = []
    t0 = time.time()

    # --- data
    if args.synthetic:
        clip_dir = os.path.join(args.out, "synthetic_clips")
        make_synthetic_clips(clip_dir, n_clips=args.synthetic_clips, seconds=args.synthetic_seconds, seed=args.seed)
        log(f"-- synthetic clips written to {clip_dir}")
    else:
        clip_dir = args.data
    log(f"-- loading clips from {clip_dir}")
    clips = load_clips(clip_dir)
    if not clips:
        log("no usable clips (need motion.parquet + audio.wav with face_valid runs >= 1 s). Try --synthetic.")
        return 2
    summary = summarise(clips)
    is_real = any(c.source != "synthetic" for c in clips)
    log(f"   {summary['n_clips']} clips, {summary['valid_minutes']} valid minutes ({summary['valid_minutes_with_audio']} with audio), "
        f"sources {summary['sources']}")
    if summary["valid_minutes_with_audio"] < 3.0 and is_real:
        flags.append(f"TINY_DATA: only {summary['valid_minutes_with_audio']} valid minutes with audio; numbers below are not representative")
    train_clips, val_clips, split_info = split_clips(clips, args.val_frac, args.seed)
    if split_info.get("leaky"):
        flags.append("LEAKY_SPLIT: a single subject/clip was split in time; held-out numbers overstate generalisation")
    if not val_clips:
        flags.append("NO_HELDOUT_SPLIT: too little data to hold anything out; metrics are computed on the training clips")
        val_clips = train_clips
    log(f"   split by {split_info['mode']}: train {len(train_clips)} / val {len(val_clips)} clips; held out {split_info['held_out_groups']}")
    stats = compute_stats(train_clips)
    timing["data"] = time.time() - t0

    # --- stage 1
    t1 = time.time()
    log("-- stage 1: VQ tokenizer")
    segs_tr, segs_va = vq_segments(train_clips, stats), vq_segments(val_clips, stats)
    vq, vq_hist, vq_base = train_vq(segs_tr, segs_va, stats, args, device, log)
    seqs_tr = run_code_sequences(train_clips, stats, vq.encode)
    seqs_va = run_code_sequences(val_clips, stats, vq.encode)
    used_tr, ppl_tr = vq.quantizer.usage(torch.as_tensor(np.concatenate(seqs_tr)))
    used_va, ppl_va = vq.quantizer.usage(torch.as_tensor(np.concatenate(seqs_va))) if seqs_va else (0, 0.0)
    log(f"   codes used on whole training runs: {used_tr}/{args.n_codes} (perplexity {ppl_tr:.1f}); held-out {used_va} ({ppl_va:.1f})")
    if is_real and used_tr < MIN_USED_CODES_REAL:
        flags.append(f"VQ_CODEBOOK_UNDERUSED: {used_tr} < {MIN_USED_CODES_REAL} codes used on real data")
    vq_info = {"n_codes": args.n_codes, "dim": args.vq_dim, "n_train_segments": int(len(segs_tr)), "n_val_segments": int(len(segs_va)),
               "best_val_mae": min((h.get("val_mae", float("inf")) for h in vq_hist), default=None), "baseline_val_mae": vq_base,
               "used_codes_train": used_tr, "perplexity_train": ppl_tr, "used_codes_val": used_va, "perplexity_val": ppl_va,
               "history": vq_hist}
    vq.save(os.path.join(args.out, "vq.pt"), {"stats": {k: v.tolist() for k, v in stats.items()}, "info": {k: v for k, v in vq_info.items() if k != "history"}})
    timing["vq"] = time.time() - t1

    # --- stage 2
    t2 = time.time()
    log("-- stage 2: audio -> codes")
    ch_tr, ch_va = a2m_chunks(train_clips, stats, vq.encode), a2m_chunks(val_clips, stats, vq.encode)
    if len(ch_tr["codes"]) == 0:
        log("no audio-aligned chunks to train on (do the clips have audio.wav?)")
        return 2
    a2m, a2m_hist = train_a2m(ch_tr, ch_va, args, device, log)
    bigram = BigramPrior(args.n_codes).fit(seqs_tr)
    bigram_logp, unigram = bigram.log_probs(), bigram.unigram()
    a2m_info = {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
                "n_train_chunks": int(len(ch_tr["codes"])), "n_val_chunks": int(len(ch_va["codes"])), "history": a2m_hist}
    info = {"channels": list(MODEL_CHANNELS), "stats": {k: v.tolist() for k, v in stats.items()}, "unigram": unigram.tolist(),
            "training": {"command": command, "device": device, "data": summary, "split": split_info,
                         "vq": {k: v for k, v in vq_info.items() if k != "history"},
                         "a2m": {k: v for k, v in a2m_info.items() if k != "history"}}}
    a2m.save(os.path.join(args.out, "a2m.pt"), {"bigram_logp": torch.from_numpy(bigram_logp),
                                                 "bigram_counts": torch.from_numpy(bigram.counts.astype(np.float32)),
                                                 "unigram": torch.from_numpy(unigram), "info": info["training"]})
    with open(os.path.join(args.out, "model_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    timing["a2m"] = time.time() - t2

    # --- retrieval baseline
    t3 = time.time()
    index = RetrievalIndex.build(train_clips, max_windows=args.max_retrieval, seed=args.seed)
    index.save(args.out, "retrieval")
    log(f"-- retrieval index: {len(index)} windows (from {index.meta.get('n_source_windows', len(index))})")
    timing["retrieval"] = time.time() - t3

    # --- metrics
    t4 = time.time()
    log("-- held-out metrics")
    model = MotionModel.load(args.out, device)
    profiles = []
    for r in args.robots:
        p = r if os.path.exists(r) else os.path.join(robots_root(), r)
        try:
            profiles.append(load_profile(p))
        except Exception as e:  # noqa: BLE001
            log(f"   could not load robot profile {r}: {e}")
    ev = evaluate(model, index, val_clips, profiles, seed=args.seed, temperature=args.temperature,
                  bigram_weight=args.bigram_weight, max_runs=args.max_eval_runs or None)
    if "verdict" in ev and not ev["verdict"]["model_beats_shuffled_audio_on_beat_recall"]:
        flags.append("MODEL_DOES_NOT_BEAT_SHUFFLED_AUDIO: retrieval ships as the default backend")
    timing["metrics"] = time.time() - t4

    metrics = {"command": command, "device": device, "flags": flags, "data": summary, "split": split_info,
               "n_train_clips": len(train_clips), "n_val_clips": len(val_clips), "vq": vq_info, "a2m": a2m_info,
               "eval": ev, "timing_s": timing}

    # --- export
    if not args.no_export:
        t5 = time.time()
        log(f"-- export -> {args.export_dir}")
        model.info = info
        rep = export_bundle(model, index, args.export_dir, metrics)
        for k in ("a2m", "vq_decoder"):
            r = rep[k]
            log(f"   {os.path.basename(r['path'])}: {r['bytes'] / 1e6:.2f} MB ({r['exporter']}), max abs diff vs torch {r['max_abs_diff']:.2e} -> {'OK' if r['ok'] else 'MISMATCH'}")
            if not r["ok"]:
                flags.append(f"ONNX_MISMATCH: {k} max abs diff {r['max_abs_diff']:.2e}")
        if "retrieval" in rep:
            log(f"   retrieval index {rep['retrieval']['n_windows']} windows, {rep['retrieval']['bytes'] / 1e6:.2f} MB; bundle {rep['total_bytes'] / 1e6:.2f} MB")
        metrics["export"] = rep
        timing["export"] = time.time() - t5
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
