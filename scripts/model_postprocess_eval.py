"""Post-hoc generation-side rows for a finished checkpoint: utterance-final settle, head_pitch
floor and amplitude (1.0 / 1.2) on every held-out speaker at the checkpoint's shipped
sampling. Writes ``postprocess_rows`` + ``postprocess`` into <ckpt>/metrics.json, records the
shipped defaults in <ckpt>/model_info.json, and regenerates REPORT.md.

    python scripts/model_postprocess_eval.py --ckpt checkpoints/v2a --data data/clips \
        --holdout kende_interview_2014 obama_2015_02_07 --exclude sd_rapper_interview cbp_vlog_day2 \
        [--settle-s 0.5 --pitch-floor -3 --amplitude 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.model.data import load_clips, split_clips  # noqa: E402
from animacy.model.infer import MotionModel  # noqa: E402
from animacy.model.metrics import postprocess_rows  # noqa: E402
from animacy.model.retrieval import RetrievalIndex  # noqa: E402
from animacy.model.train import write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/clips")
    ap.add_argument("--holdout", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--cache-dir", default="checkpoints/feature_cache")
    ap.add_argument("--settle-s", type=float, default=0.5)
    ap.add_argument("--pitch-floor", type=float, default=-3.0)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-index", action="store_true",
                    help="rebuild the retrieval index from the training clips (adds the intent arousal fields)")
    ap.add_argument("--include-holdout", action="store_true",
                    help="index every clip, hold-outs included (the SHIPPED index; never use for metrics)")
    ap.add_argument("--index-out", default="", help="write the rebuilt index here instead of <ckpt> (e.g. web/models)")
    ap.add_argument("--index-only", action="store_true", help="rebuild the index and stop (no metric rows)")
    ap.add_argument("--speaker-cap", type=float, default=0.4, help="max share of index windows for one speaker")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--max-retrieval", type=int, default=5000, help="5000 windows x 1500 B = 7.5 MB")
    a = ap.parse_args()
    clips = [c for c in load_clips(a.data, verbose=False, cache_dir=a.cache_dir or None) if c.name not in set(a.exclude)]
    train, val, _ = split_clips(clips, holdout=a.holdout)
    print(f"{len(clips)} clips; train {len(train)} / held out {[c.name for c in val]}", flush=True)
    model = MotionModel.load(a.ckpt, a.device)
    if a.rebuild_index:
        from animacy.model.data import apply_speaker_cap, load_speaker_index, mirror_clip

        src = list(clips) if a.include_holdout else list(train)
        if a.speaker_cap > 0:
            src, ci = apply_speaker_cap(src, load_speaker_index(a.data), a.speaker_cap)
            print(f"speaker cap {a.speaker_cap}: minutes before {sum(ci['minutes_before'].values()):.1f} -> effective "
                  f"{sum(ci['minutes_effective'].values()):.1f}; keep-probabilities {ci['weights'] or 'none needed'}", flush=True)
        if not a.no_mirror:
            src = src + [mirror_clip(c) for c in src]
        index = RetrievalIndex.build(src, max_windows=a.max_retrieval, seed=a.seed)
        out_dir = a.index_out or a.ckpt
        b, j = index.save(out_dir, "retrieval")
        print(f"rebuilt retrieval index -> {out_dir}: {len(index)} windows (from {index.meta.get('n_source_windows')}), "
              f"{(os.path.getsize(b) + os.path.getsize(j)) / 1e6:.2f} MB, hold-outs {'INCLUDED' if a.include_holdout else 'excluded'}, "
              f"arousal fields present", flush=True)
        if a.index_only:
            return 0
    index = RetrievalIndex.load(os.path.join(a.ckpt, "retrieval.json"))
    m_path = os.path.join(a.ckpt, "metrics.json")
    metrics = json.load(open(m_path, encoding="utf-8"))
    sampling = metrics.get("sampling") or model.info.get("sampling") or {}
    print(f"held out {[c.name for c in val]}; sampling {sampling}", flush=True)
    rows = postprocess_rows(model, index, val, sampling, seed=a.seed)
    for r in rows:
        print(f"settle {r['settle_s']}s pitch_floor {r['pitch_floor']} amp {r['amplitude']}: " + "; ".join(
            f"{cond} beat {v['beat_recall']:.3f} vs {v['beat_recall_shuffled']:.3f} (min margin {v['margin_min']:+.3f}) "
            f"still {v['stillness']:.3f} (gt {v['gt_stillness']:.3f}) W1 {v['w1_relative_mean']:.3f}" for cond, v in r["conditions"].items()), flush=True)
    pp = {"settle_s": a.settle_s, "pitch_floor": a.pitch_floor, "amplitude": a.amplitude}
    metrics["postprocess_rows"] = rows
    metrics["postprocess"] = pp
    json.dump(metrics, open(m_path, "w", encoding="utf-8"), indent=1, default=float)
    info_path = os.path.join(a.ckpt, "model_info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    info["postprocess"] = pp
    json.dump(info, open(info_path, "w", encoding="utf-8"), indent=1)
    write_report(os.path.join(a.ckpt, "REPORT.md"), metrics, metrics.get("command", ""))
    print(f"wrote {m_path}, {info_path}, REPORT.md (shipped postprocess defaults {pp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
