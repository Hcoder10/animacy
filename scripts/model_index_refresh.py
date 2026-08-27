"""One-command refresh of the retrieval index from ANY clips directory.

    python scripts/model_index_refresh.py --clips data/clips \
        --server-out checkpoints/v2a --web-out web/models \
        [--holdout kende_interview_2014 obama_2015_02_07 --eval] [--exclude ...] [--speaker-cap 0.4]

Builds, from every usable clip under ``--clips`` (face_valid runs >= 1 s with audio):

* the SERVER-SIDE index -> ``<server-out>/retrieval.{json,bin}``: every window kept (no
  cap; brute-force cosine search is fine up to a few hundred thousand windows), so
  ``animacy say --checkpoint <server-out>`` uses the whole corpus. RAM when loaded:
  ~1.3 KB/window for float32 keys + 0.8 KB/window for float16 motion (``motion_fp16``),
  e.g. 200k windows = ~430 MB; a query is one [N, 330] matvec per 0.5 s hop (~50 ms at 200k
  on one core). Disk = 1.5 KB/window (float16).
* the WEB index -> ``<web-out>/retrieval.{json,bin}``: the same corpus uniformly
  subsampled to ``--max-web-windows`` (5000 = 7.5 MB) for the browser.

Both carry the intent fields (per-window arousal, still_then_move, energy breakpoints),
the 40 % per-speaker cap (``_index.json`` speaker field, name-prefix fallback) and
mirrored copies of every window (a left-right mirror of a human is a valid human).
Speaker names are read from ``<clips>/_index.json`` ("speaker"); the two hold-outs
are INCLUDED in the shipped indexes (more human windows) and EXCLUDED from the
metrics index that ``--eval`` builds in memory to measure retrieval on them.
``--reexport-json --ckpt <dir>`` rewrites ``<web-out>/model.json`` only (fp16, AR-only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.model.data import (apply_speaker_cap, load_clips, load_speaker_index, mirror_clip,  # noqa: E402
                                split_clips, summarise)
from animacy.model.retrieval import RetrievalIndex  # noqa: E402


def index_status(clips_dir: str):
    """clip name -> status from ``_index.json`` ({} when the fetcher wrote none)."""
    path = os.path.join(clips_dir, "_index.json")
    if not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    items = d if isinstance(d, list) else next((v for v in d.values() if isinstance(v, list)), [])
    return {str(it["name"]): str(it.get("status", "kept")) for it in items if isinstance(it, dict) and it.get("name")}


def build(clips, speakers, cap, mirror, max_windows, seed):
    src = list(clips)
    info = None
    if cap > 0:
        src, info = apply_speaker_cap(src, speakers, cap)
    if mirror:
        src = src + [mirror_clip(c) for c in src]
    return RetrievalIndex.build(src, max_windows=max_windows, seed=seed), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--cache-dir", default="checkpoints/feature_cache")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--all-status", action="store_true",
                    help="also index clips whose _index.json status is not 'kept' (default: kept only; unlisted clips are kept)")
    ap.add_argument("--holdout", nargs="*", default=[], help="clips excluded from the METRICS index only (--eval)")
    ap.add_argument("--speaker-cap", type=float, default=0.4, help="max share of windows for one speaker (0 = off)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--server-out", default="", help="dir for the full (uncapped) index, e.g. checkpoints/v2a")
    ap.add_argument("--web-out", default="", help="dir for the browser index, e.g. web/models")
    ap.add_argument("--max-web-windows", type=int, default=5000)
    ap.add_argument("--eval", action="store_true", help="retrieval-only held-out metrics with an index that excludes --holdout")
    ap.add_argument("--ckpt", default="checkpoints/v2a", help="checkpoint used for --eval (tokenizer / stats) and --reexport-json")
    ap.add_argument("--reexport-json", action="store_true", help="rewrite <web-out>/model.json only (fp16, archs ar)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    status = index_status(a.clips)
    clips = [c for c in load_clips(a.clips, verbose=False, cache_dir=a.cache_dir or None) if c.name not in set(a.exclude)]
    n_all = len(clips)
    if not a.all_status and status:
        clips = [c for c in clips if status.get(c.name, "kept") == "kept"]
    if not clips:
        print("no usable clips")
        return 2
    speakers = load_speaker_index(a.clips)
    s = summarise(clips)
    print(f"{s['n_clips']} clips ({n_all - s['n_clips']} dropped by _index.json status), {s['valid_minutes']} valid min, "
          f"{len(speakers)} indexed speaker entries; features in {time.time() - t0:.0f} s", flush=True)

    report = {"clips": s["n_clips"], "dropped_by_status": n_all - s["n_clips"], "valid_minutes": s["valid_minutes"],
              "clip_names": [c.name for c in clips]}
    if a.server_out:
        idx, info = build(clips, speakers, a.speaker_cap, not a.no_mirror, 0, a.seed)
        b, j = idx.save(a.server_out, "retrieval")
        mem = RetrievalIndex.memory_estimate(len(idx))
        report["server"] = {"dir": a.server_out, "windows": len(idx), "mb_on_disk": round((os.path.getsize(b) + os.path.getsize(j)) / 1e6, 2),
                            "ram_mb_fp16_motion": round(mem["total_mb"], 1), "cap": info["weights"] if info else None}
        print(f"server index -> {a.server_out}: {len(idx)} windows, {report['server']['mb_on_disk']} MB on disk, "
              f"~{report['server']['ram_mb_fp16_motion']} MB RAM; cap weights {report['server']['cap'] or 'none needed'}", flush=True)
    if a.web_out:
        idx, info = build(clips, speakers, a.speaker_cap, not a.no_mirror, a.max_web_windows, a.seed)
        b, j = idx.save(a.web_out, "retrieval")
        report["web"] = {"dir": a.web_out, "windows": len(idx), "from_windows": idx.meta.get("n_source_windows"),
                         "mb_on_disk": round((os.path.getsize(b) + os.path.getsize(j)) / 1e6, 2)}
        print(f"web index -> {a.web_out}: {len(idx)} of {report['web']['from_windows']} windows, {report['web']['mb_on_disk']} MB", flush=True)
    if a.eval and a.holdout:
        from animacy.model.infer import MotionModel
        from animacy.model.metrics import compact, evaluate

        train, val, _ = split_clips(clips, holdout=a.holdout)
        idx, _ = build(train, speakers, a.speaker_cap, not a.no_mirror, 0, a.seed)
        model = MotionModel.load(a.ckpt, "cpu")
        rows = {}
        for c in val:
            e = evaluate(model, idx, [c], [], seed=a.seed, archs=[], verbose=False, settle_s=0.5, pitch_floor=-3.0)
            pc = compact(e)
            r, rs = pc["conditions"]["retrieval"], pc["conditions"]["retrieval_shuffled"]
            rows[c.name] = {"beat_recall": r["beat_recall"], "beat_recall_shuffled": rs["beat_recall"],
                            "margin": r["beat_recall"] - rs["beat_recall"], "beat_precision": r["beat_precision"],
                            "stillness": r["stillness"], "gt_stillness": pc["gt_stillness"], "w1_relative_mean": r["w1_relative_mean"]}
            print(f"held-out {c.name}: retrieval beat {rows[c.name]['beat_recall']:.3f} vs shuffled {rows[c.name]['beat_recall_shuffled']:.3f} "
                  f"({rows[c.name]['margin']:+.3f}), precision {rows[c.name]['beat_precision']:.3f}, stillness {rows[c.name]['stillness']:.3f} "
                  f"(gt {pc['gt_stillness']:.3f}), W1 rel {rows[c.name]['w1_relative_mean']:.3f}", flush=True)
        report["eval"] = {"index_windows": len(idx), "index_clips": len(train), "rows": rows,
                          "settings": "settle 0.5 s, pitch floor -3, amplitude 1.0, no intent bias"}
    if a.reexport_json and a.web_out:
        from animacy.model.export import main as export_main

        export_main(["--ckpt", a.ckpt, "--out", a.web_out, "--fp16", "--archs", "ar", "--json-only"])
    out = os.path.join(a.server_out or a.web_out or ".", "index_refresh.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, default=float)
    print(f"wrote {out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
