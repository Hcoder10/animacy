"""Print the corpus table from ``data/clips/_index.json`` (written by
``scripts/data_capture_batch.py index``): kept clips with stats, dropped clips
with reasons, valid-minute totals per batch, speaker/source variety.

    python scripts/data_report.py [--index data/clips/_index.json] [--all]

valid minutes = sum(face_valid * duration) / 60 over kept clips. ``--all`` also
lists dropped clips' full stats (default: name + reason only).
"""
from __future__ import annotations

import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "clips", "_index.json"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    idx = json.load(open(a.index, encoding="utf-8"))
    rows = idx["clips"]
    kept = [r for r in rows if r["status"] == "kept"]
    dropped = [r for r in rows if r["status"] != "kept"]
    g = idx.get("gate", {})
    print(f"index {idx.get('generated_at')}  gate: face_valid >= {g.get('min_face_valid')}, valid >= {g.get('min_valid_s')} s")
    print()
    hdr = f"{'clip':<34} {'dur s':>6} {'valid s':>7} {'face':>5} {'arm':>5} {'spk':>5} {'yaw sd':>6} {'pit sd':>6} {'license':<14} {'artist':<28} json"
    print("KEPT")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(kept, key=lambda r: r["name"]):
        art = (r.get("artist") or "")[:28]
        print(f"{r['name']:<34} {r['duration_s']:>6.0f} {r['valid_s']:>7.0f} {r['face_valid']:>5.2f} {r['arm_valid']:>5.2f} "
              f"{r['speaking']:>5.2f} {r['head_yaw_std']:>6.1f} {r['head_pitch_std']:>6.1f} {str(r['license']):<14} {art:<28} "
              f"{'y' if r.get('has_motion_json') else ''}".encode("ascii", "replace").decode())
    print()
    print("DROPPED / EXCLUDED")
    for r in sorted(dropped, key=lambda r: (r["status"], r["name"])):
        line = f"{r['name']:<34} [{r['status']}] {r['reason']}"
        if a.all:
            line += f"   (dur {r['duration_s']:.0f}s face {r['face_valid']:.2f} spk {r['speaking']:.2f})"
        print(line.encode("ascii", "replace").decode())
    print()
    by_batch = collections.defaultdict(lambda: {"n": 0, "dur": 0.0, "valid": 0.0})
    for r in kept:
        b = by_batch[r.get("batch", "?")]
        b["n"] += 1
        b["dur"] += r["duration_s"]
        b["valid"] += r["valid_s"]
    print("TOTALS (kept)")
    for b, v in sorted(by_batch.items()):
        print(f"  {b:<20} {v['n']:>3} clips  {v['dur'] / 60:>7.1f} min captured  {v['valid'] / 60:>7.1f} valid min")
    tot_valid = sum(r["valid_s"] for r in kept) / 60
    tot_dur = sum(r["duration_s"] for r in kept) / 60
    print(f"  {'ALL':<20} {len(kept):>3} clips  {tot_dur:>7.1f} min captured  {tot_valid:>7.1f} valid min   "
          f"({len(dropped)} dropped)")
    t = idx.get("totals", {})
    cap = g.get("speaker_cap")
    if t:
        print(f"  raw valid minutes: {t.get('raw_valid_min')}   after {cap:.0%} per-speaker cap (training-time): "
              f"{t.get('capped_valid_min')}")
    print()
    per_spk = collections.defaultdict(float)
    n_spk = collections.Counter()
    for r in kept:
        per_spk[r.get("speaker", "?")] += r["valid_s"] / 60
        n_spk[r.get("speaker", "?")] += 1
    print(f"SPEAKERS: {len(per_spk)} distinct speaker keys among kept clips (valid min, share)")
    for s, m in sorted(per_spk.items(), key=lambda kv: -kv[1]):
        print(f"  {n_spk[s]:>3} clips {m:>6.1f} min {m / tot_valid:>5.0%}  {s}".encode("ascii", "replace").decode())
    series = collections.Counter(r.get("series", "?") for r in kept)
    print("SERIES:", dict(series))
    lic = collections.Counter(str(r.get("license")) for r in kept)
    print("LICENSES:", dict(lic))
    cats = collections.Counter(str(r.get("category")) for r in kept)
    print("SOURCE CATEGORIES:", dict(cats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
