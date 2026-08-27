"""Scale the licensed talking-head corpus: enumerate Wikimedia Commons categories /
archive.org searches, license-verify every candidate with the SAME rules as
``scripts/fetch_sources.py`` (imported, not copied), download the approved ones
into ``data/raw/`` and append them to ``data/raw/sources.json`` in the fetcher's
record format (so ``animacy capture`` copies the license evidence into meta.json).

What this adds on top of fetch_sources.py (nothing there is modified):

* category enumeration (``list=categorymembers``, optional one-level subcats);
* batched ``videoinfo`` (50 titles per call) for duration / height / license so a
  200-file category costs 4 API calls instead of 200;
* polite pacing + 429 back-off on the API itself (fetch_sources only backs off on
  downloads; a burst of ~12 API calls already gets throttled);
* pre-filters that need no pixels: duration window, mime is video, a 480p (or
  >= 360p) transcode exists, title keyword blocklist (b-roll, music, montage ...);
* archive.org advancedsearch with ``licenseurl`` filters, verified per item via
  ``fetch_sources.resolve_archive`` (metadata field is the evidence, not the query).

Usage
-----
    python scripts/data_fetch_more.py plan   --out data/raw/candidates.json [--limit-per-cat 60]
    python scripts/data_fetch_more.py fetch  --plan data/raw/candidates.json --pick "File:A.webm,File:B.webm"
    python scripts/data_fetch_more.py fetch  --plan data/raw/candidates.json --max-n 40 [--allow-sa]

``plan`` writes every enumerated candidate with its license verdict (accepted or
the refusal reason) so refusals are auditable. ``fetch`` downloads only accepted
ones (optionally a hand-picked subset) and never re-downloads a file already in
``sources.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fetch_sources as fs  # noqa: E402  (the license rules live there; we only call them)

API = "https://commons.wikimedia.org/w/api.php"
_last_call = 0.0
API_GAP_S = 1.2  # Commons throttles bursts; ~1 call/s is what the API etiquette page asks for


def _paced_get_json(url: str, timeout: int = 60) -> dict:
    """Drop-in for fetch_sources._get_json with pacing + 429 back-off. fetch_sources.resolve()
    calls the module-level _get_json, so rebinding it here paces the fetcher's own resolves
    without editing fetch_sources.py (the first pass lost 50 picks to unpaced 429s)."""
    global _last_call
    for attempt in range(6):
        wait = _last_call + API_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": fs.UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 5:
                back = 20 * (attempt + 1)
                print(f"   api 429, backing off {back}s", flush=True)
                time.sleep(back)
                continue
            raise
    raise RuntimeError("api: retries exhausted")


fs._get_json = _paced_get_json


def api(params: Dict, retries: int = 5) -> dict:
    global _last_call
    params = {"format": "json", "formatversion": "2", **params}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        wait = _last_call + API_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": fs.UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                back = 20 * (attempt + 1)
                print(f"   api 429, backing off {back}s", flush=True)
                time.sleep(back)
                continue
            raise
    raise RuntimeError("api: retries exhausted")


def category_files(cat: str, subcats: int = 0, limit: int = 500) -> List[str]:
    """File titles in ``cat`` (optionally descending ``subcats`` levels)."""
    if not cat.startswith("Category:"):
        cat = "Category:" + cat
    files: List[str] = []
    subs: List[str] = []
    cont: Dict = {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "file|subcat",
                 "cmlimit": "500", **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            (files if m["ns"] == 6 else subs).append(m["title"])
        cont = d.get("continue") or {}
        if not cont or len(files) >= limit:
            break
    if subcats > 0:
        for s in subs:
            files += category_files(s, subcats - 1, limit)
    return files


TITLE_BLOCK = re.compile(
    r"b-?roll|music|song|montage|timelapse|time-lapse|drone|helicopter|aerial|flyover|trailer|"
    r"animation|animated|slideshow|highlights|compilation|ceremony|parade|concert|panel|"
    r"(\bwalk(ing)?\b)|tour|rally|protest|footage|360|silent", re.I)


def batched_videoinfo(titles: List[str]) -> Dict[str, dict]:
    """title -> videoinfo dict (url/size/mime/duration/extmetadata/derivatives), 50 per call."""
    out: Dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = api({"action": "query", "titles": "|".join(chunk), "prop": "videoinfo",
                 "viprop": "url|size|extmetadata|derivatives|mime", "vilimit": "1"})
        for p in d.get("query", {}).get("pages", []):
            vi = (p.get("videoinfo") or [None])[0]
            if vi:
                out[p["title"]] = vi
    return out


def classify_commons(title: str, vi: dict, allow_sa: bool) -> dict:
    """Same license logic as fetch_sources.resolve_commons, applied to a prefetched videoinfo."""
    ext = vi.get("extmetadata", {}) or {}
    fields = {}
    for k in ("LicenseShortName", "License", "LicenseUrl", "UsageTerms", "Copyrighted"):
        v = ext.get(k, {}).get("value")
        if v:
            fields[k] = re.sub(r"<[^>]+>", "", str(v)).strip()
    blob = " ".join(fields.get(k, "") for k in ("LicenseShortName", "License", "LicenseUrl", "UsageTerms"))
    if fields.get("Copyrighted", "").lower() == "false":
        blob += " public domain"
    ok, label = fs.classify_license(blob, allow_sa)
    best_h = 0
    for d in vi.get("derivatives", []) or []:
        if d.get("transcodekey"):
            best_h = max(best_h, int(d.get("height", 0) or 0))
    return {
        "candidate": "commons:" + title, "title": title[5:], "backend": "commons",
        "page_url": vi.get("descriptionurl"), "mime": vi.get("mime"),
        "duration_s": round(float(vi.get("duration") or 0), 1), "orig_height": vi.get("height"),
        "best_transcode_h": best_h, "orig_size_mb": round(int(vi.get("size", 0) or 0) / 1e6, 1),
        "license_ok": ok, "license": label, "license_evidence": {"api": "commons videoinfo extmetadata", **fields},
        "artist": re.sub(r"<[^>]+>", "", str(ext.get("Artist", {}).get("value", ""))).strip()[:120],
    }


def plan_commons(categories: List[Dict], allow_sa: bool, min_s: float, max_s: float, limit_per_cat: int) -> List[dict]:
    rows: List[dict] = []
    seen = set()
    for c in categories:
        cat, depth = c["cat"], int(c.get("subcats", 0))
        try:
            titles = category_files(cat, depth)
        except Exception as exc:
            print(f"  ERR listing {cat}: {type(exc).__name__}: {exc}", flush=True)
            continue
        titles = [t for t in titles if t not in seen and t.lower().endswith((".webm", ".ogv", ".mp4", ".mpg", ".mov"))]
        print(f"== {cat}: {len(titles)} video files", flush=True)
        infos = batched_videoinfo(titles[:600])
        kept = 0
        for t in titles:
            vi = infos.get(t)
            if not vi:
                continue
            seen.add(t)
            r = classify_commons(t, vi, allow_sa)
            r["category"] = cat
            r["prefilter"] = []
            if not (str(r["mime"] or "").startswith("video") or str(r["mime"] or "") == "application/ogg"):
                r["prefilter"].append("not video")
            if r["duration_s"] < min_s:
                r["prefilter"].append(f"short {r['duration_s']}s")
            if r["duration_s"] > max_s:
                r["prefilter"].append(f"long {r['duration_s']}s")
            if r["best_transcode_h"] < 360 and int(r["orig_height"] or 0) < 360:
                r["prefilter"].append("no >=360p rendition")
            if TITLE_BLOCK.search(r["title"]):
                r["prefilter"].append("title blocklist")
            r["eligible"] = bool(r["license_ok"] and not r["prefilter"])
            rows.append(r)
            if r["eligible"]:
                kept += 1
            if kept >= limit_per_cat:
                break
        print(f"   eligible {kept}", flush=True)
    return rows


def plan_archive(queries: List[str], allow_sa: bool, min_s: float, max_s: float, limit: int) -> List[dict]:
    rows: List[dict] = []
    for q in queries:
        params = {"q": q, "fl[]": ["identifier", "title", "licenseurl", "runtime", "creator"], "rows": str(limit),
                  "output": "json", "sort[]": "downloads desc"}
        url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params, doseq=True)
        try:
            docs = fs._get_json(url).get("response", {}).get("docs", [])
        except Exception as exc:
            print(f"  ERR archive search {q!r}: {exc}", flush=True)
            continue
        print(f"== archive {q!r}: {len(docs)} hits", flush=True)
        for d in docs:
            ident = d.get("identifier")
            if not ident:
                continue
            time.sleep(1.0)
            try:
                rec = fs.resolve_archive(ident, 120.0, allow_sa)  # per-item metadata is the evidence
            except Exception as exc:
                rows.append({"candidate": "archive:" + ident, "backend": "archive", "title": d.get("title"),
                             "license_ok": False, "license": f"ERR {exc}", "eligible": False, "prefilter": []})
                continue
            r = {"candidate": "archive:" + ident, "backend": "archive", "title": str(d.get("title", ident))[:120],
                 "page_url": f"https://archive.org/details/{ident}", "license_ok": rec.get("license_ok", False),
                 "license": rec.get("license"), "license_evidence": rec.get("license_evidence"),
                 "artist": str(d.get("creator", ""))[:120], "duration_s": None, "prefilter": [], "query": q}
            if rec.get("error"):
                r["prefilter"].append(rec["error"])
            dur = rec.get("duration_s")
            try:
                if dur and ":" in str(dur):
                    h, m, s = ([0, 0] + [float(x) for x in str(dur).split(":")])[-3:]
                    dur = h * 3600 + m * 60 + s
                dur = float(dur) if dur else None
            except ValueError:
                dur = None
            r["duration_s"] = dur
            if dur is not None and dur < min_s:
                r["prefilter"].append(f"short {dur}s")
            if dur is not None and dur > max_s:
                r["prefilter"].append(f"long {dur}s")
            if TITLE_BLOCK.search(r["title"]):
                r["prefilter"].append("title blocklist")
            r["eligible"] = bool(r["license_ok"] and not r["prefilter"])
            rows.append(r)
    return rows


DEFAULT_COMMONS = [
    {"cat": "Barack Obama weekly video addresses", "subcats": 1},
    {"cat": "Donald Trump 2017 weekly video addresses"},
    {"cat": "Weekly address of the President of the United States", "subcats": 1},
    {"cat": "Interview videos"},
    {"cat": "Scientist interview videos"},
    {"cat": "Scientist interview videos about a study"},
    {"cat": "Videos of Wikipedians"},
    {"cat": "WikiDonne video interviews"},
    {"cat": "Vlog videos"},
    {"cat": "Voice of America videos"},
    {"cat": "Videos of lectures"},
]
DEFAULT_ARCHIVE = [
    'mediatype:movies AND (licenseurl:*publicdomain* OR licenseurl:*licenses/by/*) AND (title:interview) AND NOT collection:*television*',
]


def cmd_plan(a) -> int:
    rows = plan_commons(a.commons or DEFAULT_COMMONS, a.allow_sa, a.min_s, a.max_s, a.limit_per_cat)
    if not a.no_archive:
        rows += plan_archive(a.archive or DEFAULT_ARCHIVE, a.allow_sa, a.min_s, a.max_s, a.archive_rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False)
    elig = [r for r in rows if r["eligible"]]
    refused = [r for r in rows if not r["license_ok"]]
    print(f"\n{len(rows)} candidates, {len(elig)} eligible, {len(refused)} license-refused -> {a.out}")
    for r in elig:
        print(f"  {r['duration_s'] or '?':>7} s  {r['license']:<14} {r['candidate']}".encode("ascii", "replace").decode())
    return 0


def cmd_fetch(a) -> int:
    rows = json.load(open(a.plan, encoding="utf-8"))
    picks = [p.strip() for p in a.pick.split(",") if p.strip()] if a.pick else None
    if a.pick_file:  # JSON list; titles may contain commas, so --pick's comma split is unsafe for them
        picks = (picks or []) + [p for p in json.load(open(a.pick_file, encoding="utf-8")) if p]
    if picks:
        want = [r for r in rows if r["candidate"] in picks or r["candidate"].split(":", 1)[1] in picks]
        known = {r["candidate"] for r in want} | {r["candidate"].split(":", 1)[1] for r in want}
        for pk in picks:  # picks outside the plan (e.g. archive:<id>) are still license-verified by resolve()
            if pk not in known:
                want.append({"candidate": pk if ":" in pk and not pk.startswith("File:") else "commons:" + pk,
                             "eligible": True, "category": "manual pick"})
    else:
        want = [r for r in rows if r["eligible"]][: a.max_n]
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "sources.json")
    index: List[Dict] = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
    by_file = {r["file"]: r for r in index}
    have_ids = {r.get("id") for r in index}
    n_ok = 0
    for r in want:
        cand = r["candidate"]
        if cand.split(":", 1)[1] in have_ids:
            print(f"  have {cand}".encode("ascii", "replace").decode())
            continue
        try:
            rec = fs.resolve(cand, a.max_mb, a.allow_sa)  # re-verify at fetch time, exactly like the fetcher
        except Exception as exc:
            print(f"  ERR  {cand}: {type(exc).__name__}: {exc}".encode("ascii", "replace").decode())
            continue
        if rec.get("error") or not rec.get("license_ok"):
            print(f"  SKIP {cand}: {rec.get('error') or rec.get('license')}".encode("ascii", "replace").decode())
            continue
        print(f"== {rec['title']}  [{rec['license']}]".encode("ascii", "replace").decode(), flush=True)
        time.sleep(2.0)
        path = fs.fetch(rec, out_dir, a.max_mb)
        if not path:
            print("   FAILED", flush=True)
            continue
        size_mb = os.path.getsize(path) / 1e6
        probe = fs.probe_video(path)
        print(f"   {size_mb:.1f} MB {probe}", flush=True)
        if probe.get("width") and int(probe["width"]) < a.min_width:
            print(f"   too narrow ({probe['width']} < {a.min_width}); not indexed", flush=True)
            continue
        entry = {
            "file": os.path.basename(path), "title": rec["title"], "url": rec["url"], "page_url": rec["page_url"],
            "license": rec["license"], "license_evidence": rec["license_evidence"], "artist": rec.get("artist", ""),
            "backend": rec["backend"], "id": rec["id"], "size_mb": round(size_mb, 1), **probe,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "category": r.get("category") or r.get("query"),
        }
        by_file[entry["file"]] = entry
        n_ok += 1
        with open(index_path, "w", encoding="utf-8") as fh:  # write after every file so a crash loses nothing
            json.dump(list(by_file.values()), fh, indent=2, ensure_ascii=False)
    print(f"fetched {n_ok}; {index_path} has {len(by_file)} entries")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "candidates.json"))
    pl.add_argument("--commons", type=json.loads, default=None, help='JSON list of {"cat":..,"subcats":n}')
    pl.add_argument("--archive", nargs="*", default=None, help="archive.org advancedsearch queries")
    pl.add_argument("--no-archive", action="store_true")
    pl.add_argument("--archive-rows", type=int, default=40)
    pl.add_argument("--limit-per-cat", type=int, default=60)
    pl.add_argument("--min-s", type=float, default=75.0)
    pl.add_argument("--max-s", type=float, default=1500.0)
    pl.add_argument("--allow-sa", action="store_true")
    pl.set_defaults(fn=cmd_plan)
    ft = sub.add_parser("fetch")
    ft.add_argument("--plan", default=os.path.join(ROOT, "data", "raw", "candidates.json"))
    ft.add_argument("--pick", default="", help="comma list of candidate ids (File:... or archive identifiers)")
    ft.add_argument("--pick-file", default="", help="JSON list of candidate ids (safe for titles with commas)")
    ft.add_argument("--max-n", type=int, default=30)
    ft.add_argument("--out-dir", default=os.path.join(ROOT, "data", "raw"))
    ft.add_argument("--max-mb", type=float, default=120.0)
    ft.add_argument("--min-width", type=int, default=480)
    ft.add_argument("--allow-sa", action="store_true")
    ft.set_defaults(fn=cmd_fetch)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
