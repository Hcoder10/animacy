"""Capture every not-yet-captured video in ``data/raw/`` into ``data/clips/<name>/``,
N at a time, then gate the results and write ``data/clips/_index.json``.

    python scripts/data_capture_batch.py run   [--jobs 3] [--duration 480] [--only a,b]
    python scripts/data_capture_batch.py index [--min-face-valid 0.6] [--min-valid-s 60]

``run`` shells out to ``python -m animacy.cli capture --source <video> -o <clip>
--neutral-seconds 0 --duration <s>`` (never --preview), skipping any raw file that
some clip's meta.json already names as ``source_path``. Stdout/err of each job goes
to ``data/logs/capture_<name>.log``.

``index`` reads every clip's meta.json (stats are computed by capture, this only
reads them), applies the gate (face_valid >= min and face_valid * duration >= min
seconds), and writes ``_index.json``: one row per clip with status kept/dropped +
reason, all validity stats, source url, license and evidence. Clips that were in
the corpus before this batch are tagged ``batch: "initial"`` (from
``--initial-names``), the rest ``batch: "scale-2026-08-26"``; ``scripts/data_report.py``
prints both totals.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CLIPS = os.path.join(ROOT, "data", "clips")
LOGS = os.path.join(ROOT, "data", "logs")
PY = sys.executable
VIDEO_EXT = (".mp4", ".mpg", ".mpeg", ".ogv", ".webm", ".mkv", ".mov")
INITIAL = ["cbp_vlog_day2", "kende_interview_2014", "obama_2014_09_13", "obama_2015_02_07",
           "royal_society_cloke", "sd_rapper_interview"]


def clip_name(raw_file: str, title: str = "") -> str:
    """Short stable clip directory name from the raw file name."""
    stem = os.path.splitext(os.path.basename(raw_file))[0]
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_President_(\w+?)_s_Weekly_Address", stem)
    if m:
        return f"{m.group(4).lower()}_wa_{m.group(1)}_{m.group(2)}_{m.group(3)}"
    m = re.match(r"(?:Weekly_Address_)?(\d{2})_(\d{2})_(\d{4})", stem)
    if m:
        return f"wa_{m.group(3)}_{m.group(1)}_{m.group(2)}"
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:48].rstrip("_") or "clip"


def speaker_key(r: Dict) -> str:
    """Who is on screen (for training-time subsampling). Series with one known speaker map to
    that person; everything else falls back to the uploader/artist slug, then the clip name
    (one uploader can be several interviewees, so this over-merges rather than under-merges)."""
    t = (r.get("title") or "") + " " + (r.get("category") or "")
    tl = t.lower()
    if "obama" in tl:
        return "obama"
    if "reagan" in tl:
        return "reagan"
    if "trump" in tl:
        return "trump"
    if "weekly conversation" in tl or "biden" in tl:
        return "biden"
    if "radio address (1996)" in tl:
        return "clinton"
    art = (r.get("artist") or "").strip()
    if art and art.lower() not in ("unknown authorunknown author", "unknown author"):
        return re.sub(r"[^a-z0-9]+", "_", art.lower()).strip("_")[:40]
    return r.get("name") or "?"


def series_key(r: Dict) -> str:
    t = ((r.get("title") or "") + " " + (r.get("category") or "")).lower()
    if "weekly conversation" in t:
        return "weekly_conversation"
    if "weekly address" in t or "radio address" in t or "weeklyaddress" in t or "address to nasa" in t:
        return "weekly_address"
    if "vlog" in t or "video blog" in t:
        return "vlog"
    if "lecture" in t:
        return "lecture"
    if "voice of america" in t or "voa" in t:
        return "voa_interview"
    if "interview" in t or "intervju" in t or "entrevista" in t or "teadlane" in t:
        return "interview"
    return "other"


def capped_minutes(per_speaker_s: Dict[str, float], cap: float) -> float:
    """Seconds left after subsampling so no speaker exceeds ``cap`` of the total:
    T = sum_i min(v_i, cap*T), solved by fixed-point iteration from T = sum v_i."""
    T = sum(per_speaker_s.values())
    for _ in range(200):
        T2 = sum(min(v, cap * T) for v in per_speaker_s.values())
        if abs(T2 - T) < 1e-6:
            break
        T = T2
    return T


def captured_sources() -> Dict[str, str]:
    """raw file basename -> clip name, for every clip that already exists."""
    out = {}
    if not os.path.isdir(CLIPS):
        return out
    for n in os.listdir(CLIPS):
        mp = os.path.join(CLIPS, n, "meta.json")
        if os.path.exists(mp):
            try:
                sp = json.load(open(mp, encoding="utf-8")).get("source_path") or ""
                if sp:
                    out[os.path.basename(sp)] = n
            except Exception:
                pass
    return out


def run_one(video: str, name: str, duration: float) -> Dict:
    out = os.path.join(CLIPS, name)
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"capture_{name}.log")
    cmd = [PY, "-m", "animacy.cli", "capture", "--source", video, "-o", out, "--neutral-seconds", "0",
           "--duration", str(duration)]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    return {"name": name, "video": video, "rc": rc, "wall_s": round(time.time() - t0, 1), "log": log}


def cmd_run(a) -> int:
    have = captured_sources()
    only = set(x for x in a.only.split(",") if x) if a.only else None
    sources = json.load(open(os.path.join(RAW, "sources.json"), encoding="utf-8"))
    titles = {r["file"]: r.get("title", "") for r in sources}
    todo = []
    for f in sorted(os.listdir(RAW)):
        if not f.lower().endswith(VIDEO_EXT) or f in have:
            continue
        if f not in titles:
            print(f"  skip {f}: not in sources.json (no license record)")
            continue
        name = clip_name(f, titles[f])
        if only and name not in only and f not in only:
            continue
        if os.path.exists(os.path.join(CLIPS, name, "motion.parquet")):
            print(f"  skip {f}: clip {name} exists")
            continue
        todo.append((os.path.join(RAW, f), name))
    print(f"{len(todo)} videos to capture with {a.jobs} workers (cap {a.duration}s each)", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, v, n, a.duration): n for v, n in todo}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            st = "ok " if r["rc"] == 0 else f"rc={r['rc']}"
            print(f"  [{len(results)}/{len(todo)}] {st} {r['name']} {r['wall_s']}s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; failures: {[r['name'] for r in results if r['rc']]}")
    return 0


def cmd_index(a) -> int:
    sources = {r["file"]: r for r in json.load(open(os.path.join(RAW, "sources.json"), encoding="utf-8"))}
    initial = set(a.initial_names.split(","))
    rows: List[Dict] = []
    for n in sorted(os.listdir(CLIPS)):
        mp = os.path.join(CLIPS, n, "meta.json")
        if not os.path.exists(mp) or not os.path.exists(os.path.join(CLIPS, n, "motion.parquet")):
            continue
        m = json.load(open(mp, encoding="utf-8"))
        st = m.get("stats", {})
        dur = st.get("n_frames", 0) / float(m.get("rate_hz", 30.0))
        fv = float(st.get("face_valid_frac", 0.0))
        valid_s = fv * dur
        src = sources.get(os.path.basename(m.get("source_path") or ""), {})
        reasons = []
        if not m.get("license") or m.get("license") == "UNKNOWN":
            reasons.append("no license record")
        if fv < a.min_face_valid:
            reasons.append(f"face_valid {fv:.0%} < {a.min_face_valid:.0%}")
        if valid_s < a.min_valid_s:
            reasons.append(f"valid {valid_s:.0f}s < {a.min_valid_s:.0f}s")
        rows.append({
            "name": n, "status": "dropped" if reasons else "kept", "reason": "; ".join(reasons),
            "batch": "initial" if n in initial else a.batch,
            "duration_s": round(dur, 1), "valid_s": round(valid_s, 1),
            "face_valid": round(fv, 3), "arm_valid": round(float(st.get("arm_valid_frac", 0)), 3),
            "torso_valid": round(float(st.get("torso_valid_frac", 0)), 3),
            "speaking": round(float(st.get("speaking_frac", 0)), 3),
            "head_yaw_std": round(float(st.get("head_yaw_std", 0)), 2),
            "head_pitch_std": round(float(st.get("head_pitch_std", 0)), 2),
            "head_roll_std": round(float(st.get("head_roll_std", 0)), 2),
            "mouth_open_std": round(float(st.get("mouth_open_std", 0)), 3),
            "src_size": m.get("src_size"), "src_fps": m.get("src_fps"),
            "title": m.get("title"), "artist": m.get("artist"), "source_url": m.get("source_url"),
            "source_file_url": m.get("source_file_url"), "license": m.get("license"),
            "license_evidence": m.get("license_evidence"), "category": src.get("category"),
            "raw_file": os.path.basename(m.get("source_path") or ""),
            "has_motion_json": os.path.exists(os.path.join(CLIPS, n, "motion.json")),
            "captured_at": m.get("captured_at"),
        })
    # Every quality-passing clip is kept. The per-speaker cap is a TRAINING-time concern: the
    # index records speaker/series so the model agent can subsample or weight, and reports
    # what the corpus would be worth under a --speaker-cap share (water-filling, see capped_minutes).
    for r in rows:
        r["speaker"], r["series"] = speaker_key(r), series_key(r)
    kept = [r for r in rows if r["status"] == "kept"]
    per_spk: Dict[str, float] = {}
    for r in kept:
        per_spk[r["speaker"]] = per_spk.get(r["speaker"], 0.0) + r["valid_s"]
    raw_min = sum(per_spk.values()) / 60
    idx = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "gate": {"min_face_valid": a.min_face_valid, "min_valid_s": a.min_valid_s,
                    "valid_minutes_definition": "sum(face_valid * duration) / 60 over kept clips",
                    "speaker_cap": a.speaker_cap,
                    "speaker_cap_note": "applied at training time, not here; capped_valid_min is what remains "
                                        "if no speaker exceeds speaker_cap of the total (water-filling)"},
           "totals": {"kept": len(kept), "dropped": len(rows) - len(kept), "raw_valid_min": round(raw_min, 1),
                      "capped_valid_min": round(capped_minutes(per_spk, a.speaker_cap) / 60, 1),
                      "valid_min_by_speaker": {k: round(v / 60, 1) for k, v in sorted(per_spk.items(), key=lambda kv: -kv[1])}},
           "clips": rows}
    with open(os.path.join(CLIPS, "_index.json"), "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=1, ensure_ascii=False)
    print(f"{len(kept)} kept / {len(rows) - len(kept)} dropped; raw valid {raw_min:.1f} min, "
          f"after {a.speaker_cap:.0%} speaker cap {idx['totals']['capped_valid_min']} min")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--jobs", type=int, default=3)
    r.add_argument("--duration", type=float, default=480.0)
    r.add_argument("--only", default="")
    r.set_defaults(fn=cmd_run)
    i = sub.add_parser("index")
    i.add_argument("--min-face-valid", type=float, default=0.6)
    i.add_argument("--min-valid-s", type=float, default=60.0)
    i.add_argument("--initial-names", default=",".join(INITIAL))
    i.add_argument("--batch", default="scale-2026-08-26")
    i.add_argument("--speaker-cap", type=float, default=0.4,
                   help="training-time per-speaker share used only to report capped_valid_min (nothing is excluded)")
    i.set_defaults(fn=cmd_index)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
