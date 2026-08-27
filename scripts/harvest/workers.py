"""N capture workers: ``fetched`` -> chunks -> prescreen -> ``animacy capture`` -> drop rule -> ``clips/``.

    python scripts/harvest/workers.py --n 16

Each worker claims the oldest fetched item, splits it into equal chunks of <= 600 s (stream copy;
``<slug>__c01`` ...; a video <= 600 s stays one clip named ``<slug>``), prescreens every chunk with
YuNet on 16 sampled frames (>= 60 % must show exactly one face, otherwise the chunk is skipped without
spending capture time), runs::

    python -m animacy.cli capture --source <chunk> -o work/<slug>/<clip> --duration 600 --neutral-seconds 0

then applies the drop rule from ``scripts/data_capture_batch.py index`` (face_valid >= 60 % and
face_valid * duration >= 60 s, license present). Kept clips: ``audio.wav`` -> ``audio.opus``
(32 kb/s; ``ffmpeg -i audio.opus -ar 16000 -ac 1 audio.wav`` restores it), a ``harvest`` block is added
to ``meta.json``, the directory moves to ``clips/<name>``, a row goes into the clips table and
``manifest.jsonl``. Dropped/skipped chunks are deleted. The raw directory is deleted when the item is
done, so the volume only ever holds the fetch buffer plus the unpushed clips.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

import common as C

PY = sys.executable


def claim(con, worker: str) -> Optional[dict]:
    with C.tx(con):
        r = con.execute("SELECT id FROM items WHERE state='fetched' ORDER BY fetched_at LIMIT 1").fetchone()
        if not r:
            return None
        con.execute("UPDATE items SET state='capturing', worker=?, updated_at=? WHERE id=?", (worker, time.time(), r["id"]))
    return dict(con.execute("SELECT * FROM items WHERE id=?", (r["id"],)).fetchone())


def run_capture(chunk_path: str, out_dir: str, duration: float, log_path: str) -> int:
    cmd = [PY, "-m", "animacy.cli", "capture", "--source", chunk_path, "-o", out_dir, "--neutral-seconds", "0",
           "--duration", str(duration)]
    with open(log_path, "w", encoding="utf-8") as fh:
        try:
            return subprocess.run(cmd, cwd=C.REPO, stdout=fh, stderr=subprocess.STDOUT, env=C.child_env(),
                                  timeout=4 * duration + 300).returncode
        except subprocess.TimeoutExpired:
            fh.write("\nTIMEOUT\n")
            return -9


def gate(meta: dict) -> Dict:
    st = meta.get("stats", {})
    dur = st.get("n_frames", 0) / float(meta.get("rate_hz", 30.0))
    fv = float(st.get("face_valid_frac", 0.0))
    valid_s = fv * dur
    reasons = []
    if not meta.get("license") or meta.get("license") == "UNKNOWN":
        reasons.append("no license record")
    if fv < C.MIN_FACE_VALID:
        reasons.append(f"face_valid {fv:.0%} < {C.MIN_FACE_VALID:.0%}")
    if valid_s < C.MIN_VALID_S:
        reasons.append(f"valid {valid_s:.0f}s < {C.MIN_VALID_S:.0f}s")
    return {"kept": not reasons, "reason": "; ".join(reasons), "duration_s": round(dur, 1), "valid_s": round(valid_s, 1), "face_valid": round(fv, 3)}


def index_row(name: str, meta: dict, g: Dict, item: dict, chunk_i: int, start: float, length: float, pre: Dict) -> Dict:
    st = meta.get("stats", {})
    return {
        "name": name, "status": "kept", "reason": "", "batch": "harvest-large",
        "duration_s": g["duration_s"], "valid_s": g["valid_s"], "face_valid": g["face_valid"],
        "arm_valid": round(float(st.get("arm_valid_frac", 0)), 3), "torso_valid": round(float(st.get("torso_valid_frac", 0)), 3),
        "speaking": round(float(st.get("speaking_frac", 0)), 3), "face_crop_frac": round(float(st.get("face_crop_frac", 0)), 3),
        "head_yaw_std": round(float(st.get("head_yaw_std", 0)), 2), "head_pitch_std": round(float(st.get("head_pitch_std", 0)), 2),
        "head_roll_std": round(float(st.get("head_roll_std", 0)), 2), "mouth_open_std": round(float(st.get("mouth_open_std", 0)), 3),
        "src_size": meta.get("src_size"), "src_fps": meta.get("src_fps"),
        "title": meta.get("title"), "artist": meta.get("artist"), "source_url": meta.get("source_url"),
        "source_file_url": meta.get("source_file_url"), "license": meta.get("license"), "license_evidence": meta.get("license_evidence"),
        "category": item.get("query"), "source_kind": item["source_kind"], "backend": item["backend"], "item_id": item["id"],
        "channel_key": item.get("channel_key"), "channel_name": item.get("channel_name"),
        "speaker": item.get("speaker_key"), "series": item.get("series"), "language": item.get("language"), "lang_evidence": item.get("lang_evidence"),
        "chunk": chunk_i, "chunk_start_s": round(start, 1), "chunk_len_s": round(length, 1), "prescreen": pre,
        "audio": "audio.opus", "has_motion_json": False, "captured_at": meta.get("captured_at"),
    }


def process_item(con, item: dict, worker: str) -> None:
    slug = C.item_slug(item["id"], item.get("title") or "")
    raw_dir = os.path.join(C.RAW, slug)
    src = item.get("raw_path") or ""
    if not os.path.exists(src):
        C.set_state(con, item["id"], "failed", error="raw file missing")
        shutil.rmtree(raw_dir, ignore_errors=True)
        return
    probe = C.ffprobe(src)
    dur = float(probe.get("duration_s") or item.get("duration_s") or 0.0)
    if dur < C.MIN_ITEM_S:
        C.set_state(con, item["id"], "dropped", error=f"duration {dur:.0f}s")
        shutil.rmtree(raw_dir, ignore_errors=True)
        return
    plan = C.chunk_plan(dur)
    records = json.load(open(os.path.join(raw_dir, "sources.json"), encoding="utf-8"))
    base_rec = records[0]
    ext = os.path.splitext(src)[1] or ".mp4"
    chunks: List[Dict] = []
    work_dir = os.path.join(C.WORK, slug)
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(C.LOGS, "capture"), exist_ok=True)
    t_item = time.time()
    n_kept, kept_s, captured_s = 0, 0.0, 0.0
    for i, (start, length) in enumerate(plan, start=1):
        name = slug if len(plan) == 1 else f"{slug}__c{i:02d}"
        rec: Dict = {"chunk": i, "start": round(start, 1), "len": round(length, 1), "name": name}
        if len(plan) == 1:
            chunk_path = src
        else:
            chunk_path = os.path.join(raw_dir, f"{name}{ext}")
            if not C.split_chunk(src, chunk_path, start, length):
                rec["result"] = "split_failed"
                chunks.append(rec)
                continue
            records.append({**base_rec, "file": os.path.basename(chunk_path)})
            with open(os.path.join(raw_dir, "sources.json"), "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=1, ensure_ascii=False)
        try:
            pre = C.prescreen(chunk_path)
        except Exception as exc:
            pre = {"ok": True, "error": f"{type(exc).__name__}: {exc}"[:120]}  # never block capture on a prescreen bug
        rec["prescreen"] = pre
        if not pre.get("ok"):
            rec["result"] = "prescreen"
            chunks.append(rec)
            if chunk_path != src:
                os.remove(chunk_path)
            continue
        out = os.path.join(work_dir, name)
        t0 = time.time()
        rc = run_capture(chunk_path, out, C.CHUNK_S, os.path.join(C.LOGS, "capture", f"{name}.log"))
        rec["capture_s"] = round(time.time() - t0, 1)
        meta_path = os.path.join(out, "meta.json")
        if rc != 0 and not os.path.exists(os.path.join(out, "motion.parquet")):
            rec["result"] = f"capture_rc{rc}"
            chunks.append(rec)
            shutil.rmtree(out, ignore_errors=True)
            if chunk_path != src:
                os.remove(chunk_path)
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        g = gate(meta)
        captured_s += g["duration_s"]
        rec.update({k: g[k] for k in ("duration_s", "valid_s", "face_valid")})
        if not g["kept"]:
            rec["result"] = "dropped: " + g["reason"]
            chunks.append(rec)
            shutil.rmtree(out, ignore_errors=True)
            if chunk_path != src:
                os.remove(chunk_path)
            continue
        wav, opus = os.path.join(out, "audio.wav"), os.path.join(out, "audio.opus")
        if os.path.exists(wav):
            if C.wav_to_opus(wav, opus):
                os.remove(wav)
            else:
                rec["opus"] = "failed, wav kept"
        row = index_row(name, meta, g, item, i, start, length, pre)
        meta["harvest"] = {k: row[k] for k in ("item_id", "source_kind", "channel_key", "channel_name", "speaker", "series",
                                                "language", "lang_evidence", "chunk", "chunk_start_s", "chunk_len_s", "prescreen", "category")}
        meta["harvest"]["worker"] = worker
        meta["audio"] = {"file": "audio.opus", "codec": f"opus {C.OPUS_KBPS} kb/s mono, source 16 kHz", "sr": 16000,
                         "restore": "ffmpeg -i audio.opus -ar 16000 -ac 1 audio.wav"}
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        dst = os.path.join(C.CLIPS, name)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(out, dst)
        nbytes = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
        with C.tx(con):
            con.execute("INSERT OR REPLACE INTO clips(name, item_id, chunk, path, duration_s, valid_s, face_valid, state, bytes, captured_at, row) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (name, item["id"], i, dst, g["duration_s"], g["valid_s"], g["face_valid"], "kept", nbytes, time.time(),
                         json.dumps(row, ensure_ascii=False)))
        C.append_jsonl(C.MANIFEST, row)
        rec["result"] = "kept"
        n_kept += 1
        kept_s += g["duration_s"]
        chunks.append(rec)
        if chunk_path != src:
            os.remove(chunk_path)
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    state = "captured" if n_kept else "dropped"
    C.set_state(con, item["id"], state, n_chunks=len(plan), n_kept=n_kept, kept_s=round(kept_s, 1),
                captured_s=round(captured_s, 1), prescreen=json.dumps(chunks, ensure_ascii=False), captured_at=time.time(),
                raw_path=None)
    wall = time.time() - t_item
    C.log(f"[{worker}] {slug}: {n_kept}/{len(plan)} chunks kept, {kept_s / 60:.1f} min kept of {dur / 60:.1f}, "
          f"{wall:.0f}s wall ({captured_s / max(wall, 1):.2f}x realtime)")


def worker_loop(worker: str, stop: threading.Event) -> None:
    con = C.db()
    paused = False
    while not stop.is_set():
        free = C.disk_free_gb()
        if free < C.CAPTURE_MIN_FREE_GB:
            if not paused:
                C.log(f"[{worker}] disk {free:.1f} GB free; paused")
                C.event(con, "disk_pause", f"{worker} paused at {free:.1f} GB")
                paused = True
            time.sleep(60)
            continue
        paused = False
        item = claim(con, worker)
        if item is None:
            time.sleep(15)
            continue
        try:
            process_item(con, item, worker)
        except Exception as exc:
            C.log(f"[{worker}] error on {item['id']}: {type(exc).__name__}: {exc}")
            C.set_state(con, item["id"], "failed", error=f"capture: {type(exc).__name__}: {exc}"[:300])
            shutil.rmtree(os.path.join(C.RAW, C.item_slug(item["id"], item.get("title") or "")), ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--once", action="store_true", help="process one item in the main thread and exit (smoke test)")
    a = p.parse_args(argv)
    C.ensure_dirs()
    con = C.db()
    # items left 'capturing' by a crashed process go back to fetched (raw files are still there) or fail
    for r in con.execute("SELECT id, raw_path FROM items WHERE state='capturing'").fetchall():
        if r["raw_path"] and os.path.exists(r["raw_path"]):
            con.execute("UPDATE items SET state='fetched' WHERE id=?", (r["id"],))
        else:
            con.execute("UPDATE items SET state='failed', error='raw lost during capture' WHERE id=?", (r["id"],))
    shutil.rmtree(C.WORK, ignore_errors=True)
    os.makedirs(C.WORK, exist_ok=True)
    C.kv_set(con, "workers_n", str(a.n))
    if a.once:
        item = claim(con, "w00")
        if not item:
            C.log("nothing fetched")
            return 1
        process_item(con, item, "w00")
        return 0
    stop = threading.Event()
    threads = [threading.Thread(target=worker_loop, args=(f"w{i:02d}", stop), daemon=True) for i in range(a.n)]
    for t in threads:
        t.start()
        time.sleep(1.0)
    C.log(f"{a.n} workers running")
    try:
        while True:
            time.sleep(30)
            C.kv_set(con, "workers_heartbeat", str(time.time()))
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
