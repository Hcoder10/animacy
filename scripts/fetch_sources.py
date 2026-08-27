"""Fetch short, license-verified talking-head videos into ``data/raw/``.

Ported from reachy-duplex ``scripts/fetch_video_sources.py`` and extended to
Wikimedia Commons. The rule is enforced in code, not remembered per download:

* a file is fetched only if its *machine-readable* license metadata says
  public domain / CC0 / CC-BY (``--allow-sa`` additionally admits CC-BY-SA);
* anything carrying **ND** (no derivatives) or **NC** is refused;
* **missing license metadata is refused** — a search engine describing an item
  as "public domain" is not evidence; the metadata field is. reachy-duplex hit
  exactly this case (Frost/Nixon: no ``licenseurl`` at all) and this check is
  what caught it.

Every accepted download is recorded in ``data/raw/sources.json`` with the url,
license, the *metadata field* the license came from (evidence), title, and
the local file. ``animacy capture`` copies that record into the clip's
``meta.json``.

Backends (a candidate is ``"<backend>:<id>"``):

* ``commons:File:Name.webm``  — Wikimedia Commons ``videoinfo`` API
  (``extmetadata.LicenseShortName`` / ``License`` / ``LicenseUrl``). Uses the
  480p transcode when one exists so files stay small.
* ``archive:<identifier>``     — archive.org metadata API (``metadata.licenseurl``
  / ``metadata.rights``); picks the smallest video file over 3 MB.
* ``ytdlp:<url>``              — any yt-dlp-supported page; accepted only when
  the info-json ``license`` field explicitly says Creative Commons Attribution
  or public domain. (No default candidates use this backend.)

Usage:
    python scripts/fetch_sources.py --list                # verify licenses only
    python scripts/fetch_sources.py --out-dir data/raw    # download approved
    python scripts/fetch_sources.py --candidate commons:File:Foo.webm --out-dir data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

UA = "animacy-fetch/0.1 (research dataset capture; contact via github.com/Hcoder10)"
MAX_MB_DEFAULT = 100.0
VIDEO_EXT = (".mp4", ".mpg", ".mpeg", ".ogv", ".webm", ".mkv", ".mov")

# Curated: single frontal speaker, 1-5 min, PD or CC-BY. Breadth over length.
CANDIDATES = [
    # US federal government works: public domain (Commons extmetadata says "Public domain").
    "commons:File:2015-02-07 President Obama's Weekly Address.webm",
    "commons:File:2014-09-13 President Obama's Weekly Address.webm",
    "commons:File:Video Blog- CBP Super Bowl LII Countdown to Kickoff - Day 2.webm",
    # Institutional uploads that are usually CC-BY on Commons (verified at fetch time, else refused).
    "commons:File:Interview on extreme weather with physical geographer Hannah Cloke – The Royal Society.webm",
    "commons:File:Internet Hall of Fame 2014 Michael Kende interview.webm",
    # archive.org public-domain interviews (small files).
    "archive:Chuck_San_Diego_LocalRapper",
    "archive:coffeehouse-Forum-November-2004",
]

# Accept / refuse markers, checked against a lower-cased blob of the license fields.
REFUSED_MARKERS = ("-nd", "by-nd", "nc-nd", "-nc", "by-nc", "noncommercial", "non-commercial", "no derivative")
PD_MARKERS = ("public domain", "publicdomain", "cc0", "pd-usgov", "pd-us", "creativecommons.org/publicdomain")
BY_MARKERS = ("cc-by-", "cc by ", "cc-by ", "licenses/by/", "attribution")
SA_MARKERS = ("by-sa", "cc by-sa", "licenses/by-sa/", "sharealike", "share alike")


def classify_license(blob: str, allow_sa: bool = False) -> Tuple[bool, str]:
    """(accepted, spdx-ish label). The blob is all license fields joined, lower-cased."""
    b = blob.lower()
    if not b.strip():
        return False, "NONE"
    if any(m in b for m in REFUSED_MARKERS):
        return False, "refused:ND/NC"
    if any(m in b for m in SA_MARKERS):
        return (True, "CC-BY-SA-4.0") if allow_sa else (False, "refused:SA (use --allow-sa)")
    if any(m in b for m in PD_MARKERS):
        return True, "CC0-1.0" if "cc0" in b or "zero" in b else "Public Domain"
    if any(m in b for m in BY_MARKERS):
        m = re.search(r"by[/ -]?(\d\.\d)", b)
        return True, f"CC-BY-{m.group(1) if m else '4.0'}"
    return False, f"refused:unrecognised ({blob.strip()[:80]})"


def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _download(url: str, dest: str, max_mb: float, retries: int = 4) -> bool:
    tmp = dest + ".part"
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
                    if total > max_mb * 1e6:
                        print(f"   aborting: exceeds {max_mb:.0f} MB")
                        fh.close()
                        os.remove(tmp)
                        return False
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            if exc.code == 429 and attempt < retries - 1:
                wait = 15 * (attempt + 1)  # Commons throttles bursts; back off politely
                print(f"   429 throttled, retrying in {wait}s")
                time.sleep(wait)
                continue
            print(f"   download failed: HTTPError {exc.code}")
            return False
        except Exception as exc:  # network errors: leave nothing partial behind
            print(f"   download failed: {type(exc).__name__}: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
    return False


def probe_video(path: str) -> Dict:
    """width/height/fps/duration via ffprobe (empty dict if ffprobe is unavailable)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=60)
        j = json.loads(out.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return {"width": s.get("width"), "height": s.get("height"), "fps": round(fps, 3),
                "duration_s": round(float(j.get("format", {}).get("duration", 0) or 0), 2)}
    except Exception:
        return {}


# ---------------------------------------------------------------- Wikimedia Commons
def resolve_commons(title: str, max_mb: float, allow_sa: bool) -> Optional[Dict]:
    if not title.startswith("File:"):
        title = "File:" + title
    q = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "videoinfo",
        "viprop": "url|size|extmetadata|derivatives|mime", "format": "json"})
    data = _get_json("https://commons.wikimedia.org/w/api.php?" + q)
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    info = (page.get("videoinfo") or [None])[0]
    if not info:
        return {"error": "no videoinfo (missing file?)"}
    ext = info.get("extmetadata", {}) or {}
    fields = {}
    for k in ("LicenseShortName", "License", "LicenseUrl", "UsageTerms", "Copyrighted"):
        v = ext.get(k, {}).get("value")
        if v:
            fields[k] = re.sub(r"<[^>]+>", "", str(v)).strip()
    # "Copyrighted: False" is Commons' PD marker; include it in the evidence blob.
    blob = " ".join(fields.get(k, "") for k in ("LicenseShortName", "License", "LicenseUrl", "UsageTerms"))
    if fields.get("Copyrighted", "").lower() == "false":
        blob += " public domain"
    ok, label = classify_license(blob, allow_sa)
    # Prefer the 480p transcode (MediaPipe face landmarks want >= ~100 px faces; 360p is
    # marginal on a wide shot), else the closest available >= 360p. The original is often
    # a 1080p multi-hundred-MB file.
    choice = None
    for d in info.get("derivatives", []) or []:
        key = str(d.get("transcodekey", ""))
        h = int(d.get("height", 0) or 0)
        if not key or h < 360:
            continue
        score = abs(h - 480) + (0 if h >= 480 else 1)
        if choice is None or score < choice[0]:
            choice = (score, d, h)
    if choice is not None:
        choice = (choice[2], choice[1])
    if choice is not None:
        url, height = choice[1]["src"], choice[0]
        size_mb = None  # transcode size unknown until fetched; enforced during download
    else:
        url, height, size_mb = info["url"], int(info.get("height", 0) or 0), int(info.get("size", 0) or 0) / 1e6
    return {
        "backend": "commons", "id": title, "title": title[5:],
        "page_url": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
        "url": url, "height": height, "size_mb": size_mb, "duration_s": info.get("duration"),
        "license_ok": ok, "license": label,
        "license_evidence": {"api": "commons videoinfo extmetadata", **fields},
        "artist": re.sub(r"<[^>]+>", "", str(ext.get("Artist", {}).get("value", ""))).strip(),
    }


# ---------------------------------------------------------------- archive.org
def resolve_archive(identifier: str, max_mb: float, allow_sa: bool) -> Optional[Dict]:
    meta = _get_json(f"https://archive.org/metadata/{identifier}")
    md = meta.get("metadata", {}) or {}
    fields = {k: str(md[k]) for k in ("licenseurl", "rights") if md.get(k)}
    blob = " ".join(fields.values())
    ok, label = classify_license(blob, allow_sa)
    best = None
    for f in meta.get("files", []) or []:
        name = str(f.get("name", ""))
        if not name.lower().endswith(VIDEO_EXT):
            continue
        try:
            size_mb = int(f.get("size", 0)) / 1e6
        except (TypeError, ValueError):
            continue
        if size_mb < 3 or size_mb > max_mb:
            continue
        if best is None or size_mb < best[0]:
            best = (size_mb, f)
    if best is None:
        return {"backend": "archive", "id": identifier, "license_ok": ok, "license": label,
                "error": f"no video file between 3 and {max_mb:.0f} MB",
                "license_evidence": {"api": "archive.org metadata", **fields}}
    size_mb, f = best
    return {
        "backend": "archive", "id": identifier, "title": str(md.get("title", identifier)),
        "page_url": f"https://archive.org/details/{identifier}",
        "url": f"https://archive.org/download/{identifier}/{urllib.parse.quote(str(f['name']))}",
        "size_mb": size_mb, "height": None, "duration_s": f.get("length"),
        "license_ok": ok, "license": label,
        "license_evidence": {"api": "archive.org metadata", **fields},
        "artist": str(md.get("creator", "")),
    }


# ---------------------------------------------------------------- yt-dlp (generic)
def resolve_ytdlp(url: str, max_mb: float, allow_sa: bool) -> Optional[Dict]:
    cmd = [sys.executable, "-m", "yt_dlp", "--dump-single-json", "--no-warnings", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"backend": "ytdlp", "id": url, "error": out.stderr.strip()[-200:]}
    info = json.loads(out.stdout)
    lic = str(info.get("license") or "")
    ok, label = classify_license(lic, allow_sa)
    return {
        "backend": "ytdlp", "id": url, "title": info.get("title", url), "page_url": info.get("webpage_url", url),
        "url": url, "size_mb": None, "height": None, "duration_s": info.get("duration"),
        "license_ok": ok, "license": label,
        "license_evidence": {"api": "yt-dlp info-json", "license": lic or "NONE"},
        "artist": str(info.get("uploader", "")),
    }


def resolve(candidate: str, max_mb: float, allow_sa: bool) -> Dict:
    backend, _, ident = candidate.partition(":")
    if backend == "commons":
        return resolve_commons(ident, max_mb, allow_sa)
    if backend == "archive":
        return resolve_archive(ident, max_mb, allow_sa)
    if backend == "ytdlp":
        return resolve_ytdlp(ident, max_mb, allow_sa)
    return {"backend": backend, "id": ident, "error": f"unknown backend {backend!r}"}


def safe_name(s: str) -> str:
    s = re.sub(r"\.(webm|ogv|mp4|mkv|mov|mpg|mpeg)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:60] or "video"


def fetch(rec: Dict, out_dir: str, max_mb: float) -> Optional[str]:
    if rec["backend"] == "ytdlp":
        dest_tpl = os.path.join(out_dir, safe_name(rec["title"]) + ".%(ext)s")
        cmd = [sys.executable, "-m", "yt_dlp", "-f", "best[height<=480]/bestvideo[height<=480]+bestaudio/best",
               "--no-warnings", "-o", dest_tpl, rec["url"]]
        if subprocess.run(cmd).returncode != 0:
            return None
        for ext in VIDEO_EXT:
            p = dest_tpl.replace("%(ext)s", ext[1:])
            if os.path.exists(p):
                return p if os.path.getsize(p) <= max_mb * 1e6 else None
        return None
    ext = os.path.splitext(urllib.parse.urlparse(rec["url"]).path)[1] or ".webm"
    dest = os.path.join(out_dir, safe_name(rec["title"]) + ext)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"   exists: {dest}")
        return dest
    return dest if _download(rec["url"], dest, max_mb) else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.path.join("data", "raw"))
    p.add_argument("--max-mb", type=float, default=MAX_MB_DEFAULT)
    p.add_argument("--list", action="store_true", help="verify licenses only, download nothing")
    p.add_argument("--allow-sa", action="store_true", help="also accept CC-BY-SA")
    p.add_argument("--candidate", action="append", default=[], help="backend:id (repeatable); default = curated list")
    p.add_argument("--min-width", type=int, default=320, help="reject downloads narrower than this (landmark quality)")
    a = p.parse_args(argv)

    cands = a.candidate or CANDIDATES
    os.makedirs(a.out_dir, exist_ok=True)
    approved: List[Dict] = []
    for c in cands:
        try:
            rec = resolve(c, a.max_mb, a.allow_sa)
        except Exception as exc:
            print(f"  ERR  {c}: {type(exc).__name__}: {exc}")
            continue
        if rec.get("error"):
            print(f"  SKIP {c}: {rec['error']}")
            continue
        if not rec["license_ok"]:
            print(f"  SKIP {c}: {rec['license']}  evidence={rec['license_evidence']}")
            continue
        sz = f"{rec['size_mb']:.0f} MB" if rec.get("size_mb") else "size unknown (transcode)"
        print(f"  OK   {c}\n       {rec['license']}  [{sz}, {rec.get('height')}p, {rec.get('duration_s')}s]"
              f"\n       evidence: {rec['license_evidence']}")
        approved.append(rec)
    print(f"\n{len(approved)}/{len(cands)} approved")
    if a.list:
        return 0

    index_path = os.path.join(a.out_dir, "sources.json")
    index: List[Dict] = []
    if os.path.exists(index_path):
        index = json.load(open(index_path, encoding="utf-8"))
    by_file = {r["file"]: r for r in index}
    for n, rec in enumerate(approved):
        print(f"== {rec['title']}")
        if n:
            time.sleep(3.0)  # do not burst the host
        path = fetch(rec, a.out_dir, a.max_mb)
        if not path:
            print("   FAILED")
            continue
        size_mb = os.path.getsize(path) / 1e6
        probe = probe_video(path)
        print(f"   {size_mb:.1f} MB -> {path}  {probe}")
        if probe.get("width") and int(probe["width"]) < a.min_width:
            print(f"   too small for face landmarks (width {probe['width']} < {a.min_width}); kept on disk, not indexed")
            continue
        entry = {
            "file": os.path.basename(path), "title": rec["title"], "url": rec["url"], "page_url": rec["page_url"],
            "license": rec["license"], "license_evidence": rec["license_evidence"], "artist": rec.get("artist", ""),
            "backend": rec["backend"], "id": rec["id"], "size_mb": round(size_mb, 1), **probe,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        by_file[entry["file"]] = entry
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(list(by_file.values()), fh, indent=2, ensure_ascii=False)
    print(f"wrote {index_path} ({len(by_file)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
