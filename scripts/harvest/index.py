"""Rebuild the ``_index.json``-style manifest for the harvest from the clips table (which outlives the
local clip directories: pushed clips are deleted from disk, their rows stay).

    python scripts/harvest/index.py [--out <ROOT>/_index.json] [--dump-queue]

Same shape as ``data/clips/_index.json`` (generated_at, gate, totals, clips) plus per-language /
per-source / per-series / per-license minute tables and the push state of every clip. ``--dump-queue``
also writes ``queue.jsonl`` (one line per item with its state) next to it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from typing import Dict, List, Optional

import common as C


def rows_from_db(con) -> List[Dict]:
    out = []
    for r in con.execute("SELECT name, state, shard, pushed_at, row FROM clips ORDER BY captured_at"):
        row = json.loads(r["row"])
        row["push_state"] = r["state"]
        row["shard"] = r["shard"]
        out.append(row)
    return out


def build(con, rows: Optional[List[Dict]] = None) -> Dict:
    from data_capture_batch import capped_minutes

    rows = rows_from_db(con) if rows is None else rows
    per_spk: Dict[str, float] = collections.defaultdict(float)
    per_lang: Dict[str, float] = collections.defaultdict(float)
    per_src: Dict[str, float] = collections.defaultdict(float)
    per_series: Dict[str, float] = collections.defaultdict(float)
    per_lic: Dict[str, float] = collections.defaultdict(float)
    per_chan: Dict[str, float] = collections.defaultdict(float)
    for r in rows:
        per_spk[r.get("speaker") or "?"] += r["valid_s"]
        per_lang[r.get("language") or "?"] += r["duration_s"]
        per_src[r.get("source_kind") or "?"] += r["duration_s"]
        per_series[r.get("series") or "?"] += r["duration_s"]
        per_lic[str(r.get("license"))] += r["duration_s"]
        per_chan[r.get("channel_key") or "?"] += r["duration_s"]
    kept_s = sum(r["duration_s"] for r in rows)
    valid_s = sum(r["valid_s"] for r in rows)
    st = {r["state"]: (r["n"], r["h"] or 0.0) for r in con.execute("SELECT state, COUNT(*) n, SUM(duration_s)/3600 h FROM items GROUP BY state")} if con is not None else {}
    top_chan = sorted(per_chan.items(), key=lambda kv: -kv[1])[:20]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gate": {"min_face_valid": C.MIN_FACE_VALID, "min_valid_s": C.MIN_VALID_S,
                 "valid_minutes_definition": "sum(face_valid * duration) / 60 over kept clips",
                 "prescreen": f">= {C.PRESCREEN_MIN_FRAC:.0%} of {C.PRESCREEN_FRAMES} sampled frames show exactly one face (YuNet)",
                 "channel_cap": f"no channel above {C.CHANNEL_CAP_SHARE:.0%} of kept seconds (floor {C.CHANNEL_CAP_FLOOR_H} h), applied at fetch time",
                 "speaker_cap": 0.4, "speaker_cap_note": "training-time; capped_valid_min is what remains if no speaker exceeds 40 % (water-filling)",
                 "audio": f"audio.opus {C.OPUS_KBPS} kb/s (ffmpeg -i audio.opus -ar 16000 -ac 1 audio.wav restores 16 kHz wav)"},
        "totals": {"kept": len(rows), "kept_hours": round(kept_s / 3600, 2), "valid_hours": round(valid_s / 3600, 2),
                   "raw_valid_min": round(valid_s / 60, 1), "capped_valid_min": round(capped_minutes(dict(per_spk), 0.4) / 60, 1),
                   "pushed": sum(1 for r in rows if r["push_state"] == "pushed"),
                   "items_by_state": {k: {"n": v[0], "hours": round(v[1], 1)} for k, v in st.items()},
                   "speakers": len(per_spk),
                   "hours_by_language": {k: round(v / 3600, 2) for k, v in sorted(per_lang.items(), key=lambda kv: -kv[1])},
                   "hours_by_source": {k: round(v / 3600, 2) for k, v in sorted(per_src.items(), key=lambda kv: -kv[1])},
                   "hours_by_series": {k: round(v / 3600, 2) for k, v in sorted(per_series.items(), key=lambda kv: -kv[1])},
                   "hours_by_license": {k: round(v / 3600, 2) for k, v in sorted(per_lic.items(), key=lambda kv: -kv[1])},
                   "top_channels_share": {k: round(v / max(kept_s, 1), 3) for k, v in top_chan},
                   "valid_min_by_speaker": {k: round(v / 60, 1) for k, v in sorted(per_spk.items(), key=lambda kv: -kv[1])[:200]}},
        "clips": rows,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(C.ROOT, "_index.json"))
    p.add_argument("--dump-queue", action="store_true")
    a = p.parse_args(argv)
    con = C.db()
    idx = build(con)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, a.out)
    t = idx["totals"]
    C.log(f"{t['kept']} kept clips, {t['kept_hours']} h kept ({t['valid_hours']} h valid), {t['speakers']} speaker keys -> {a.out}")
    if a.dump_queue:
        qp = os.path.join(os.path.dirname(a.out), "queue.jsonl")
        with open(qp, "w", encoding="utf-8") as fh:
            for r in con.execute("SELECT * FROM items"):
                d = dict(r)
                d.pop("prescreen", None)
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        C.log(f"wrote {qp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
