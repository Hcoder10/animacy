"""Paced, resumable downloads: ``queued`` -> ``fetched`` (or ``refused`` / ``failed``).

    python scripts/harvest/fetch.py [--loop] [--buffer 36] [--max-mb 150]

One download at a time (bandwidth is not the bottleneck, YouTube's tolerance is). Keeps at most
``--buffer`` items in the fetched-but-not-captured state, pauses when the volume has less than
``FETCH_MIN_FREE_GB`` free, and on HTTP 429 / "confirm you're not a bot" backs YouTube off
(5 -> 10 -> 20 -> 30 min, probe before resuming) while Commons / archive.org items keep flowing;
every incident goes to the events table. ~300 YouTube downloads/h from one IP tripped the bot check
after ~1.5 h on 2026-08-27; the default pacing is now ~150/h.

License verification happens HERE for YouTube (the crawl only saw the flat listing):
  * ytsearch: ``--match-filter "license ~= creative commons attribution"`` skips anything else before
    a byte is downloaded, and the info-json ``license`` is re-checked with
    ``fetch_sources.classify_license`` afterwards (must be CC-BY);
  * usgov: the info-json ``channel_id`` must equal the allowlisted channel's id;
  * commons / archive: ``fetch_sources.resolve()`` re-reads the metadata fields at fetch time (as the
    fetcher and data_fetch_more do), then ``fetch_sources.fetch()`` downloads.
The 5 % per-channel cap (floor 3 h while the corpus is small) is applied when picking: a channel whose
kept + in-flight seconds exceed the cap is passed over until the total grows.

480p, <= 150 MB: AVC 480p when it fits, 360p for items longer than 25 min. Each item lands in
``raw/<slug>/`` with a ``sources.json`` in the fetcher's record format so ``animacy capture`` copies
the license evidence into meta.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional

import common as C
from common import fs

PY = sys.executable
RATE_PATTERNS = ("429", "Too Many Requests", "not a bot", "Sign in to confirm", "rate-limit", "rate limit")


PROBE_URL = "https://www.youtube.com/watch?v=c8_BIamLESg"   # a CC-BY talk used only for --simulate health probes


class Pacer:
    """YouTube-only back-off: 5 -> 10 -> 20 -> 30 min (cap), escalation forgotten after 2 h of success.
    While YouTube is blocked, non-YouTube backends keep fetching; before YouTube resumes, one cheap
    ``--simulate`` probe must pass, otherwise the block is extended without spending a download."""

    def __init__(self) -> None:
        self.backoff = 0
        self.blocked_until = 0.0
        self.last_ok = time.time()

    def penalty(self, con, detail: str) -> None:
        if time.time() - self.last_ok > 7200:
            self.backoff = 0
        self.backoff = min(self.backoff + 1, 4)
        wait = min(300 * (2 ** (self.backoff - 1)), 1800)
        self.blocked_until = time.time() + wait
        C.event(con, "ratelimit", detail[-400:])
        C.log(f"   youtube blocked; backing off {wait / 60:.0f} min: {detail[-160:]}")

    def ok(self) -> None:
        self.last_ok = time.time()
        self.backoff = 0

    def youtube_open(self, con) -> bool:
        if time.time() < self.blocked_until:
            return False
        if self.backoff == 0:
            return True
        cmd = [PY, "-m", "yt_dlp", *C.YTDLP_COMMON, "--no-warnings", "--simulate", "--print", "%(id)s", PROBE_URL]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=C.child_env(), encoding="utf-8", errors="replace")
        blob = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and not any(p in blob for p in RATE_PATTERNS):
            C.log("   youtube probe ok; resuming")
            C.event(con, "youtube_resume", f"after {self.backoff} back-offs")
            return True
        self.penalty(con, "probe: " + blob)
        return False


def channel_caps(con) -> Dict[str, float]:
    """channel_key -> seconds already kept + in flight; plus the cap value under key '__cap__'."""
    tot = con.execute("SELECT COALESCE(SUM(duration_s),0) s FROM clips").fetchone()["s"]
    cap = max(C.CHANNEL_CAP_SHARE * tot, C.CHANNEL_CAP_FLOOR_H * 3600)
    per: Dict[str, float] = {"__cap__": cap}
    for r in con.execute("SELECT i.channel_key k, COALESCE(SUM(c.duration_s),0) s FROM clips c JOIN items i ON i.id=c.item_id GROUP BY i.channel_key"):
        per[r["k"] or ""] = per.get(r["k"] or "", 0.0) + r["s"]
    for r in con.execute("SELECT channel_key k, COALESCE(SUM(duration_s),0) s FROM items WHERE state IN ('fetching','fetched','capturing') GROUP BY channel_key"):
        per[r["k"] or ""] = per.get(r["k"] or "", 0.0) + 0.6 * r["s"]  # in flight, assume ~60 % survives
    return per


def pick(con, allow_youtube: bool = True) -> Optional[dict]:
    caps = channel_caps(con)
    cap = caps["__cap__"]
    rows = con.execute("SELECT * FROM items WHERE state='queued' AND (part % ?) = ? ORDER BY priority DESC, queued_at LIMIT 400",
                       (C.PART_N, C.PART_K)).fetchall()
    if not allow_youtube:
        rows = con.execute("SELECT * FROM items WHERE state='queued' AND (part % ?) = ? AND backend != 'ytdlp' "
                           "ORDER BY priority DESC, queued_at LIMIT 400", (C.PART_N, C.PART_K)).fetchall()
    for r in rows:
        if caps.get(r["channel_key"] or "", 0.0) >= cap:
            continue
        with C.tx(con):
            cur = con.execute("UPDATE items SET state='fetching', updated_at=?, attempts=COALESCE(attempts,0)+1 "
                              "WHERE id=? AND state='queued'", (time.time(), r["id"]))
            if cur.rowcount == 1:
                return dict(r)
    return None


def sources_record(item: dict, filename: str) -> dict:
    ev = item.get("license_evidence")
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except json.JSONDecodeError:
            ev = {"raw": ev}
    return {"file": filename, "title": item.get("title"), "url": item.get("url"), "page_url": item.get("page_url"),
            "license": item.get("license"), "license_evidence": ev, "artist": item.get("artist") or item.get("channel_name"),
            "backend": item["backend"], "id": item["id"], "category": item.get("query"),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def write_sources_json(raw_dir: str, records: list) -> None:
    with open(os.path.join(raw_dir, "sources.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=1, ensure_ascii=False)


def fetch_youtube(con, item: dict, raw_dir: str, max_mb: float, pacer: Pacer) -> Optional[str]:
    """Download + verify one YouTube item. Returns the path or None (state already set on failure)."""
    long_item = (item.get("duration_s") or 0) > 25 * 60
    h = 360 if long_item else 480
    fmt = (f"bv*[height<={h}][vcodec~='^(avc1|h264)']+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/w")
    if item["source_kind"] == "usgov":
        mfilter = f"channel_id = '{item['expected_channel_id']}'"
    else:
        mfilter = "license ~= '(?i)creative commons attribution'"
    out_tpl = os.path.join(raw_dir, "source.%(ext)s")
    cmd = [PY, "-m", "yt_dlp", *C.YTDLP_COMMON, "--no-warnings", "--no-playlist", "--no-progress", "--match-filter", mfilter,
           "--write-info-json", "--no-write-playlist-metafiles", "-f", fmt, "--max-filesize", f"{int(max_mb)}M",
           "--merge-output-format", "mp4", "--ffmpeg-location", C.bin_path("ffmpeg"), "-o", out_tpl,
           "--sleep-requests", "0.7", "--retries", "3", "--fragment-retries", "3", "--socket-timeout", "30",
           "--no-continue", "--no-part", item["url"]]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=C.child_env(), encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        C.set_state(con, item["id"], "failed", error="yt-dlp timeout")
        return None
    blob = (r.stdout or "") + "\n" + (r.stderr or "")
    if any(p in blob for p in RATE_PATTERNS):
        shutil.rmtree(raw_dir, ignore_errors=True)
        C.set_state(con, item["id"], "queued")  # not the item's fault; retry after the back-off
        pacer.penalty(con, blob)
        return None
    files = [p for p in glob.glob(os.path.join(raw_dir, "source.*")) if not p.endswith(".json")]
    info_path = os.path.join(raw_dir, "source.info.json")
    if not files and ("does not pass filter" in blob or "skipping" in blob.lower()):
        C.set_state(con, item["id"], "refused", error="match-filter: " + (blob.strip().splitlines() or [""])[-1][:200],
                    license_ok=0)
        shutil.rmtree(raw_dir, ignore_errors=True)
        return None
    if "exceeds max file size" in blob or "File is larger than max-filesize" in blob:
        C.set_state(con, item["id"], "failed", error="exceeds max filesize")
        shutil.rmtree(raw_dir, ignore_errors=True)
        return None
    if r.returncode != 0 or not files or not os.path.exists(info_path):
        C.set_state(con, item["id"], "failed", error=("rc=%d " % r.returncode) + (r.stderr or "").strip()[-300:])
        shutil.rmtree(raw_dir, ignore_errors=True)
        return None
    info = json.load(open(info_path, encoding="utf-8"))
    lic = str(info.get("license") or "")
    ok, label = fs.classify_license(lic, allow_sa=False)
    if ok and "creative commons attribution" in lic.lower():
        label = "CC-BY-3.0"  # YouTube's only CC option is CC BY 3.0; classify_license defaults to 4.0 when no version is in the string
    evidence = {"api": "yt-dlp info-json", "license": lic or "NONE", "channel_id": info.get("channel_id"),
                "channel": info.get("channel") or info.get("uploader"), "uploader_id": info.get("uploader_id"),
                "upload_date": info.get("upload_date"), "webpage_url": info.get("webpage_url")}
    if item["source_kind"] == "usgov":
        prev = json.loads(item.get("license_evidence") or "{}")
        if info.get("channel_id") != item.get("expected_channel_id"):
            C.set_state(con, item["id"], "refused", error=f"channel id {info.get('channel_id')} != allowlisted {item.get('expected_channel_id')}", license_ok=0)
            shutil.rmtree(raw_dir, ignore_errors=True)
            return None
        evidence = {**prev, **evidence, "basis": "PD-USGov"}
        label = "Public Domain"
    else:
        if not ok or not label.startswith("CC-BY"):
            C.set_state(con, item["id"], "refused", error=f"license {lic!r} -> {label}", license_ok=0, license_evidence=json.dumps(evidence))
            shutil.rmtree(raw_dir, ignore_errors=True)
            return None
    language, lev = C.guess_language(info.get("title") or item.get("title") or "", info, item.get("language") or "")
    small = {k: info.get(k) for k in ("id", "title", "channel_id", "channel", "uploader", "uploader_id", "upload_date",
                                       "license", "language", "duration", "webpage_url", "categories", "tags", "view_count")}
    small["automatic_captions_orig"] = [k for k in (info.get("automatic_captions") or {}) if str(k).endswith("-orig")]
    with open(os.path.join(raw_dir, "info.min.json"), "w", encoding="utf-8") as fh:
        json.dump(small, fh, ensure_ascii=False)
    os.remove(info_path)
    con.execute("UPDATE items SET license=?, license_evidence=?, license_ok=1, language=?, lang_evidence=?, "
                "channel_key=COALESCE(channel_key, ?), artist=COALESCE(artist, ?) WHERE id=?",
                (label, json.dumps(evidence, ensure_ascii=False), language, lev, info.get("channel_id"),
                 info.get("channel") or info.get("uploader"), item["id"]))
    pacer.ok()
    return files[0]


def fetch_fs(con, item: dict, raw_dir: str, max_mb: float) -> Optional[str]:
    """Commons / archive via the license-verified fetcher (re-verifies the metadata at fetch time)."""
    try:
        rec = fs.resolve(item["id"], max_mb, False)
    except Exception as exc:
        C.set_state(con, item["id"], "failed", error=f"resolve: {type(exc).__name__}: {exc}"[:300])
        return None
    if rec.get("error"):
        C.set_state(con, item["id"], "failed", error=str(rec["error"])[:300])
        return None
    if not rec.get("license_ok"):
        C.set_state(con, item["id"], "refused", error=str(rec.get("license"))[:200], license_ok=0,
                    license_evidence=json.dumps(rec.get("license_evidence") or {}, ensure_ascii=False))
        return None
    path = fs.fetch(rec, raw_dir, max_mb)
    if not path:
        C.set_state(con, item["id"], "failed", error="download failed")
        return None
    con.execute("UPDATE items SET license=?, license_evidence=?, license_ok=1, artist=COALESCE(artist, ?), url=? WHERE id=?",
                (rec["license"], json.dumps(rec.get("license_evidence") or {}, ensure_ascii=False), rec.get("artist"), rec.get("url"), item["id"]))
    return path


def fetch_one(con, item: dict, max_mb: float, pacer: Pacer) -> bool:
    slug = C.item_slug(item["id"], item.get("title") or "")
    raw_dir = os.path.join(C.RAW, slug)
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)
    t0 = time.time()
    if item["backend"] == "ytdlp":
        path = fetch_youtube(con, item, raw_dir, max_mb, pacer)
    else:
        path = fetch_fs(con, item, raw_dir, max_mb)
    if not path:
        return False
    probe = C.ffprobe(path)
    if not probe.get("width") or int(probe["width"]) < 320:
        C.set_state(con, item["id"], "failed", error=f"too small: {probe}")
        shutil.rmtree(raw_dir, ignore_errors=True)
        return False
    item = dict(con.execute("SELECT * FROM items WHERE id=?", (item["id"],)).fetchone())
    write_sources_json(raw_dir, [sources_record(item, os.path.basename(path))])
    size_mb = os.path.getsize(path) / 1e6
    C.set_state(con, item["id"], "fetched", raw_path=path, size_mb=round(size_mb, 1), height=probe.get("height"),
                fps=probe.get("fps"), duration_s=probe.get("duration_s") or item.get("duration_s"), fetched_at=time.time())
    C.log(f"   fetched {slug}: {size_mb:.0f} MB {probe.get('width')}x{probe.get('height')} {probe.get('duration_s')}s in {time.time() - t0:.0f}s [{item['license']}]")
    return True


def buffer_size(con) -> int:
    return con.execute("SELECT COUNT(*) n FROM items WHERE state IN ('fetched','capturing')").fetchone()["n"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--buffer", type=int, default=36, help="max items fetched-but-not-captured")
    p.add_argument("--max-mb", type=float, default=C.MAX_FILE_MB)
    p.add_argument("--gap", type=float, default=20.0, help="mean seconds between YouTube downloads (~150/h; 8 s = ~300/h tripped the bot check)")
    p.add_argument("--n", type=int, default=0, help="stop after this many successful fetches (0 = unlimited)")
    p.add_argument("--retry-failed", action="store_true", help="re-queue failed items with < 2 attempts first")
    a = p.parse_args(argv)
    C.ensure_dirs()
    con = C.db()
    C.log(f"fetch: partition {C.PART_K}/{C.PART_N}, box {C.BOX!r}")
    if a.retry_failed:
        con.execute("UPDATE items SET state='queued', error=NULL WHERE state='failed' AND COALESCE(attempts,0) < 2")
    # anything left 'fetching' by a crashed process goes back to the queue
    con.execute("UPDATE items SET state='queued' WHERE state='fetching'")
    pacer = Pacer()
    n_ok = 0
    idle_logged = False
    while True:
        free = C.disk_free_gb()
        if free < C.FETCH_MIN_FREE_GB:
            if not idle_logged:
                C.log(f"disk {free:.1f} GB free < {C.FETCH_MIN_FREE_GB}; fetch paused")
                C.event(con, "disk_pause", f"fetch paused at {free:.1f} GB free")
                idle_logged = True
            time.sleep(60)
            continue
        if buffer_size(con) >= a.buffer:
            time.sleep(20)
            continue
        C.backfill_parts(con)
        yt_open = pacer.youtube_open(con)
        item = pick(con, allow_youtube=yt_open)
        if item is None:
            if not idle_logged:
                C.log("queue empty (or every channel at cap%s); waiting" % ("" if yt_open else ", youtube blocked"))
                idle_logged = True
            if not a.loop:
                return 0
            time.sleep(60)
            continue
        idle_logged = False
        C.log(f"== fetch [{item['source_kind']}] {item['title'][:70]!r} {item.get('duration_s') or 0:.0f}s")
        try:
            ok = fetch_one(con, item, a.max_mb, pacer)
        except Exception as exc:
            C.log(f"   error: {type(exc).__name__}: {exc}")
            C.set_state(con, item["id"], "failed", error=f"{type(exc).__name__}: {exc}"[:300])
            ok = False
        n_ok += ok
        if a.n and n_ok >= a.n:
            return 0
        if item["backend"] == "ytdlp":
            time.sleep(random.uniform(0.6 * a.gap, 1.4 * a.gap))
        else:
            time.sleep(3.0)


if __name__ == "__main__":
    raise SystemExit(main())
