"""Source -> candidates in the queue (state ``queued``).

    python scripts/harvest/crawl.py [--sources ytsearch,usgov,commons,archive] [--per-query 100]
                                    [--exclude-index data/clips/_index.json] [--loop --every-hours 12]

* ytsearch: each (query, lang) in sources.YT_QUERIES under each sort order, CC-filtered search URL,
  flat listing (title/duration/channel only; the license is verified at fetch time).
* usgov: flat listing of each allowlisted agency channel; license = Public Domain (PD-USGov basis),
  expected channel id recorded and re-checked at fetch.
* commons / archive: ``scripts/data_fetch_more.plan_commons`` / ``plan_archive`` (license verified now,
  from the metadata fields).

Dedupe: by id (url) and by title fingerprint. Skips < 60 s, > 60 min, live/upcoming, title blocklist,
and anything whose page url is already in the given local index. Per crawl round a YouTube channel
contributes at most ``--per-channel`` new items (variety); the 5 % channel cap is enforced at fetch time.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
from typing import Dict, Iterable, List, Optional, Set

import common as C
import sources as S

PY = sys.executable
BLOCK = re.compile(S.TITLE_BLOCK, re.I)


def ytdlp_flat(url: str, n: int, timeout: int = 900) -> tuple[dict, List[dict]]:
    """(playlist-level info, flat entries). Channel listings carry channel_id only at the playlist
    level; search results carry it per entry."""
    cmd = [PY, "-m", "yt_dlp", *C.YTDLP_COMMON, "--flat-playlist", "-J", "--no-warnings", "--ignore-errors",
           "--playlist-end", str(n), "--sleep-requests", "0.6", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=C.child_env(), encoding="utf-8",
                       errors="replace")
    err = (r.stderr or "").strip()
    if "429" in err or "Too Many Requests" in err or "not a bot" in err:
        raise RuntimeError("ratelimit: " + err[-300:])
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    entries = [e for e in (info.get("entries") or []) if isinstance(e, dict)]
    if not entries and err:
        C.log(f"   yt-dlp: {err[-200:]}")
    return info, entries


def existing_keys(con) -> tuple[Set[str], Set[str], Set[str]]:
    ids = {r["id"] for r in con.execute("SELECT id FROM items")}
    fps = {r["title_fp"] for r in con.execute("SELECT title_fp FROM items WHERE title_fp != ''")}
    urls = {r["page_url"] for r in con.execute("SELECT page_url FROM items WHERE page_url IS NOT NULL")}
    return ids, fps, urls


def add_candidate(con, cand: Dict, ids: Set[str], fps: Set[str], urls: Set[str], excluded: Set[str]) -> str:
    """Insert one candidate or return the reason it was not inserted."""
    cid = cand["id"]
    if cid in ids or cand.get("page_url") in urls:
        return "dup_id"
    if cand.get("page_url") in excluded:
        return "in_local_index"
    d = cand.get("duration_s")
    if d is None:
        return "no_duration"
    fp = C.title_fingerprint(cand.get("title", ""))
    if fp:
        fp = f"{fp}|{int(d // 15)}"  # same title AND same length (15 s bucket) = the same video re-uploaded
    if fp and fp in fps:
        return "dup_title"
    if d < C.MIN_ITEM_S:
        return "short"
    if d > C.MAX_ITEM_S:
        return "long"
    if BLOCK.search(cand.get("title") or ""):
        return "title_block"
    if cand.get("license_ok") is False:
        state = "refused"
    else:
        state = "queued"
    now = time.time()
    con.execute(
        "INSERT INTO items(id, backend, source_kind, url, page_url, title, title_fp, duration_s, channel_key, "
        "channel_name, expected_channel_id, speaker_key, series, language, lang_evidence, query, license, "
        "license_evidence, license_ok, state, priority, queued_at, updated_at, artist, part) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, cand["backend"], cand["source_kind"], cand["url"], cand.get("page_url"), cand.get("title", ""), fp, d,
         cand.get("channel_key"), cand.get("channel_name"), cand.get("expected_channel_id"),
         cand.get("speaker_key"), cand.get("series"), cand.get("language"), cand.get("lang_evidence"),
         cand.get("query"), cand.get("license"), json.dumps(cand.get("license_evidence") or {}, ensure_ascii=False),
         None if cand.get("license_ok") is None else int(bool(cand["license_ok"])), state,
         float(cand.get("priority", 0.0)) + random.random(), now, now, cand.get("artist"), C.item_part(cid)))
    ids.add(cid)
    if fp:
        fps.add(fp)
    if cand.get("page_url"):
        urls.add(cand["page_url"])
    return state


def yt_entry_to_candidate(e: dict, source_kind: str, query: str, lang: str, license_: Optional[str],
                          evidence: Optional[dict], expected_channel: Optional[str]) -> Optional[Dict]:
    vid = e.get("id")
    if not vid or e.get("ie_key") not in (None, "Youtube") or e.get("_type") not in (None, "url", "video"):
        return None
    if e.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        return None
    title = e.get("title") or ""
    ch = e.get("channel_id") or e.get("uploader_id") or ""
    language, lev = C.guess_language(title, None, lang)
    return {
        "id": "yt:" + vid, "backend": "ytdlp", "source_kind": source_kind,
        "url": f"https://www.youtube.com/watch?v={vid}", "page_url": f"https://www.youtube.com/watch?v={vid}",
        "title": title, "duration_s": e.get("duration"), "channel_key": ch,
        "channel_name": e.get("channel") or e.get("uploader") or "", "expected_channel_id": expected_channel,
        "speaker_key": C.speaker_key("ytdlp", ch, e.get("uploader") or "", title, vid),
        "series": C.series_key(title, query, source_kind), "language": language, "lang_evidence": lev,
        "query": query, "license": license_, "license_evidence": evidence, "license_ok": None if license_ is None else True,
        "artist": e.get("channel") or e.get("uploader") or "", "priority": 0.5 if source_kind == "usgov" else 0.0,
    }


def crawl_ytsearch(con, per_query: int, per_channel: int, excluded: Set[str], sorts: Iterable[str], queries=None) -> Dict[str, int]:
    ids, fps, urls = existing_keys(con)
    stats: Dict[str, int] = {}
    queries = list(queries or S.YT_QUERIES)
    random.shuffle(queries)
    for q, lang in queries:
        for sort in sorts:
            url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={S.YT_SP[sort]}"
            try:
                _, entries = ytdlp_flat(url, per_query)
            except RuntimeError as exc:
                C.event(con, "ratelimit", f"crawl ytsearch {q!r}: {exc}")
                C.log(f"   rate limited on search; sleeping 15 min ({exc})")
                time.sleep(900)
                continue
            per_ch: Dict[str, int] = {}
            n_new = 0
            with C.tx(con):
                for e in entries:
                    cand = yt_entry_to_candidate(e, "ytsearch", q, lang, None, None, None)
                    if not cand:
                        stats["skip_entry"] = stats.get("skip_entry", 0) + 1
                        continue
                    ck = cand["channel_key"]
                    if per_ch.get(ck, 0) >= per_channel:
                        stats["per_channel_cap"] = stats.get("per_channel_cap", 0) + 1
                        continue
                    res = add_candidate(con, cand, ids, fps, urls, excluded)
                    stats[res] = stats.get(res, 0) + 1
                    if res == "queued":
                        per_ch[ck] = per_ch.get(ck, 0) + 1
                        n_new += 1
            C.log(f"  ytsearch [{lang}] {q!r} ({sort}): {len(entries)} hits, {n_new} new")
            time.sleep(random.uniform(3, 7))
    return stats


def crawl_usgov(con, per_channel: int, excluded: Set[str]) -> Dict[str, int]:
    ids, fps, urls = existing_keys(con)
    stats: Dict[str, int] = {}
    for url, agency, note in S.USGOV_CHANNELS:
        try:
            info, entries = ytdlp_flat(url, per_channel)
        except RuntimeError as exc:
            C.event(con, "ratelimit", f"crawl usgov {agency}: {exc}")
            time.sleep(900)
            continue
        expected = info.get("channel_id") or ""
        if not entries or not expected:
            C.log(f"  usgov {agency}: no entries / no channel id (handle wrong or channel empty)")
            stats["channel_empty"] = stats.get("channel_empty", 0) + 1
            continue
        for e in entries:
            e.setdefault("channel_id", expected)
            if not e.get("channel_id"):
                e["channel_id"] = expected
        chan_ids = {e.get("channel_id") for e in entries if e.get("channel_id")} | {info.get("uploader_id")}
        evidence = {"api": "yt-dlp info-json + channel allowlist", "basis": "PD-USGov", "agency": agency,
                    "channel_url": url, "channel_id": expected, "statement": S.USGOV_BASIS}
        if note:
            evidence["caveat"] = note
        n_new = 0
        with C.tx(con):
            for e in entries:
                if e.get("channel_id") and e["channel_id"] != expected:
                    continue  # a listing should be one channel; anything else is not the agency's own upload
                cand = yt_entry_to_candidate(e, "usgov", agency, "en", "Public Domain", evidence, expected)
                if not cand:
                    continue
                cand["channel_name"] = agency
                res = add_candidate(con, cand, ids, fps, urls, excluded)
                stats[res] = stats.get(res, 0) + 1
                n_new += res == "queued"
        C.log(f"  usgov {agency}: {len(entries)} listed ({len(chan_ids)} channel ids), {n_new} new")
        time.sleep(random.uniform(3, 7))
    return stats


def crawl_commons(con, limit_per_cat: int, excluded: Set[str]) -> Dict[str, int]:
    import data_fetch_more as dfm

    ids, fps, urls = existing_keys(con)
    stats: Dict[str, int] = {}

    def batched_videoinfo_small(titles, _orig=dfm.batched_videoinfo):
        """50 Commons titles per GET overflowed the URI (HTTP 414) on long interview titles; 12 is safe."""
        out = {}
        for i in range(0, len(titles), 12):
            out.update(_orig(titles[i:i + 12]))
        return out

    dfm.batched_videoinfo = batched_videoinfo_small
    rows = []
    for cat in S.COMMONS_CATEGORIES:  # one category failing must not lose the others
        try:
            rows += dfm.plan_commons([cat], allow_sa=False, min_s=C.MIN_ITEM_S, max_s=C.MAX_ITEM_S, limit_per_cat=limit_per_cat)
        except Exception as exc:
            C.log(f"  commons {cat['cat']!r} failed: {type(exc).__name__}: {exc}")
            C.event(con, "crawl_error", f"commons {cat['cat']}: {exc}")
    with C.tx(con):
        for r in rows:
            if not r.get("license_ok"):
                stats["license_refused"] = stats.get("license_refused", 0) + 1
                cand_ok = False
            else:
                cand_ok = True
            if r.get("prefilter"):
                stats["prefilter"] = stats.get("prefilter", 0) + 1
                continue
            title = r["title"]
            language, lev = C.guess_language(title, None, "")
            art = r.get("artist") or ""
            cand = {
                "id": r["candidate"], "backend": "commons", "source_kind": "commons", "url": r["candidate"],
                "page_url": r.get("page_url"), "title": title, "duration_s": r.get("duration_s"),
                "channel_key": "commons:" + C.slug(art, 40), "channel_name": art, "artist": art,
                "speaker_key": C.speaker_key("commons", "", art, title, C.slug(title)),
                "series": C.series_key(title, r.get("category", ""), "commons"), "language": language, "lang_evidence": lev,
                "query": r.get("category"), "license": r.get("license"), "license_evidence": r.get("license_evidence"),
                "license_ok": cand_ok, "priority": 0.5,
            }
            res = add_candidate(con, cand, ids, fps, urls, excluded)
            stats[res] = stats.get(res, 0) + 1
    return stats


def crawl_archive(con, rows_per_query: int, excluded: Set[str]) -> Dict[str, int]:
    import data_fetch_more as dfm

    ids, fps, urls = existing_keys(con)
    stats: Dict[str, int] = {}
    rows = dfm.plan_archive(S.ARCHIVE_QUERIES, allow_sa=False, min_s=C.MIN_ITEM_S, max_s=C.MAX_ITEM_S, limit=rows_per_query)
    with C.tx(con):
        for r in rows:
            if r.get("prefilter"):
                stats["prefilter"] = stats.get("prefilter", 0) + 1
                continue
            title = r.get("title") or r["candidate"]
            language, lev = C.guess_language(title, None, "")
            art = r.get("artist") or ""
            cand = {
                "id": r["candidate"], "backend": "archive", "source_kind": "archive", "url": r["candidate"],
                "page_url": r.get("page_url"), "title": title, "duration_s": r.get("duration_s"),
                "channel_key": "archive:" + C.slug(art, 40), "channel_name": art, "artist": art,
                "speaker_key": C.speaker_key("archive", "", art, title, C.slug(title)),
                "series": C.series_key(title, r.get("query", ""), "archive"), "language": language, "lang_evidence": lev,
                "query": r.get("query"), "license": r.get("license"), "license_evidence": r.get("license_evidence"),
                "license_ok": bool(r.get("license_ok")), "priority": 0.3,
            }
            res = add_candidate(con, cand, ids, fps, urls, excluded)
            stats[res] = stats.get(res, 0) + 1
    return stats


def load_excluded(path: str) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    idx = json.load(open(path, encoding="utf-8"))
    out = set()
    for r in idx.get("clips", []):
        for k in ("source_url", "source_file_url"):
            if r.get(k):
                out.add(r[k])
    return out


def run_once(a) -> None:
    con = C.db()
    excluded = load_excluded(a.exclude_index)
    srcs = [s.strip() for s in a.sources.split(",") if s.strip()]
    for s in srcs:
        t0 = time.time()
        C.log(f"== crawl {s}")
        try:
            if s == "ytsearch":
                st = crawl_ytsearch(con, a.per_query, a.per_channel, excluded, a.sorts.split(","))
            elif s == "usgov":
                st = crawl_usgov(con, a.usgov_per_channel, excluded)
            elif s == "commons":
                st = crawl_commons(con, a.limit_per_cat, excluded)
            elif s == "archive":
                st = crawl_archive(con, a.archive_rows, excluded)
            else:
                C.log(f"unknown source {s}")
                continue
        except Exception as exc:
            C.log(f"  crawl {s} failed: {type(exc).__name__}: {exc}")
            C.event(con, "crawl_error", f"{s}: {exc}")
            continue
        C.event(con, "crawl", f"{s}: {json.dumps(st)} in {time.time() - t0:.0f}s")
        C.log(f"  {s}: {st}")
    q = con.execute("SELECT state, COUNT(*) n, SUM(duration_s)/3600 h FROM items GROUP BY state").fetchall()
    C.log("queue: " + ", ".join(f"{r['state']} {r['n']} ({(r['h'] or 0):.1f} h)" for r in q))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="usgov,ytsearch,commons,archive")
    p.add_argument("--per-query", type=int, default=100)
    p.add_argument("--per-channel", type=int, default=20, help="new items per YouTube channel per search round")
    p.add_argument("--usgov-per-channel", type=int, default=150)
    p.add_argument("--sorts", default="relevance,date,views")
    p.add_argument("--limit-per-cat", type=int, default=300)
    p.add_argument("--archive-rows", type=int, default=60)
    p.add_argument("--exclude-index", default=os.path.join(C.REPO, "data", "clips", "_index.json"))
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every-hours", type=float, default=12.0)
    p.add_argument("--min-queued-hours", type=float, default=200.0,
                   help="in --loop mode, only re-crawl when the queued backlog is below this many hours")
    a = p.parse_args(argv)
    C.ensure_dirs()
    while True:
        run_once(a)
        if not a.loop:
            return 0
        while True:
            time.sleep(600)
            con = C.db()
            h = (con.execute("SELECT SUM(duration_s) s FROM items WHERE state='queued'").fetchone()["s"] or 0) / 3600
            last = float(C.kv_get(con, "last_crawl", "0"))
            con.close()
            if h < a.min_queued_hours or time.time() - last > a.every_hours * 3600:
                break
        con = C.db()
        C.kv_set(con, "last_crawl", str(time.time()))
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
