"""Shared pieces of the large-scale harvest (``scripts/harvest/``).

Layout (``HARVEST_ROOT``, default ``C:\\harvest\\data`` on Windows, ``~/animacy-harvest/data`` elsewhere)::

    queue.sqlite        the queue: items (one per source video) + clips (one per kept chunk) + events
    raw/<item>/         downloaded video, its 10-min chunk files, and a sources.json the capture reads
    work/<item>/        capture output before the drop rule is applied
    clips/<name>/       kept clips (motion.parquet, audio.opus, meta.json) waiting to be pushed
    manifest.jsonl      one row per kept clip, appended at capture time (durable after local deletion)
    stage/              shard staging for push.py
    logs/               per-process logs

Item states: queued -> fetching -> fetched -> capturing -> captured | dropped
             (refused = license refused at verify time, failed = fetch/capture error, skipped = prescreen)
Clip states: kept -> pushed.

License rules are imported from ``scripts/fetch_sources.py`` (``classify_license``); nothing here
re-implements them. Speaker / series keys follow ``scripts/data_capture_batch.py``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)
import fetch_sources as fs  # noqa: E402  (license rules live there)

IS_WIN = sys.platform == "win32"
ROOT = os.environ.get("HARVEST_ROOT") or (r"C:\harvest\data" if IS_WIN else os.path.expanduser("~/animacy-harvest/data"))
BIN = os.environ.get("HARVEST_BIN") or (r"C:\harvest\bin" if IS_WIN else os.path.expanduser("~/animacy-harvest/bin"))
DB_PATH = os.path.join(ROOT, "queue.sqlite")
RAW = os.path.join(ROOT, "raw")
WORK = os.path.join(ROOT, "work")
CLIPS = os.path.join(ROOT, "clips")
STAGE = os.path.join(ROOT, "stage")
LOGS = os.path.join(ROOT, "logs")
MANIFEST = os.path.join(ROOT, "manifest.jsonl")

TARGET_HOURS = float(os.environ.get("HARVEST_TARGET_HOURS", "5000"))
CHUNK_S = 600.0            # capture --duration per chunk
MIN_ITEM_S, MAX_ITEM_S = 60.0, 3600.0
MAX_FILE_MB = 150.0
MIN_FACE_VALID, MIN_VALID_S = 0.6, 60.0     # the drop rule (same as data_capture_batch.py index)
CHANNEL_CAP_SHARE, CHANNEL_CAP_FLOOR_H = 0.05, 3.0   # no channel > 5 % of kept minutes (floor 3 h while small)
PRESCREEN_FRAMES, PRESCREEN_MIN_FRAC = 16, 0.6        # >= 60 % of sampled frames must show one dominant face
FETCH_MIN_FREE_GB, CAPTURE_MIN_FREE_GB = 6.0, 3.0     # the remote volume has ~18 GB free in total
OPUS_KBPS = 32

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".ogv", ".mpg", ".mpeg", ".m4v")


def ensure_dirs() -> None:
    for d in (ROOT, RAW, WORK, CLIPS, STAGE, LOGS):
        os.makedirs(d, exist_ok=True)


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + str(msg).encode("ascii", "replace").decode(), flush=True)


def bin_path(name: str) -> str:
    """ffmpeg/ffprobe/deno/yt-dlp: HARVEST_BIN first, then PATH."""
    for cand in (os.path.join(BIN, name + (".exe" if IS_WIN else "")), os.path.join(BIN, name)):
        if os.path.exists(cand):
            return cand
    found = shutil.which(name)
    if found:
        return found
    try:  # animacy.audio falls back to imageio-ffmpeg; mirror that
        import imageio_ffmpeg
        if name == "ffmpeg":
            return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return name


def child_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""      # the GPUs belong to another job
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("ANIMACY_MODELS_DIR", os.path.join(ROOT, "models"))
    return env


def disk_free_gb(path: str = ROOT) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except FileNotFoundError:
        return shutil.disk_usage(os.path.dirname(path)).free / 1e9


# ---------------------------------------------------------------- db
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY, backend TEXT, source_kind TEXT, url TEXT, page_url TEXT, title TEXT, title_fp TEXT,
  duration_s REAL, channel_key TEXT, channel_name TEXT, expected_channel_id TEXT, speaker_key TEXT, series TEXT,
  language TEXT, lang_evidence TEXT, query TEXT, license TEXT, license_evidence TEXT, license_ok INTEGER,
  state TEXT, error TEXT, raw_path TEXT, size_mb REAL, height INTEGER, fps REAL,
  n_chunks INTEGER, n_kept INTEGER, kept_s REAL, captured_s REAL, prescreen TEXT, worker TEXT,
  priority REAL DEFAULT 0, queued_at REAL, fetched_at REAL, captured_at REAL, updated_at REAL, artist TEXT,
  attempts INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS items_state ON items(state);
CREATE INDEX IF NOT EXISTS items_fp ON items(title_fp);
CREATE TABLE IF NOT EXISTS clips (
  name TEXT PRIMARY KEY, item_id TEXT, chunk INTEGER, path TEXT, duration_s REAL, valid_s REAL, face_valid REAL,
  state TEXT, shard TEXT, bytes INTEGER, captured_at REAL, pushed_at REAL, row TEXT
);
CREATE INDEX IF NOT EXISTS clips_state ON clips(state);
CREATE TABLE IF NOT EXISTS events (ts REAL, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def db() -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(DB_PATH, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


@contextlib.contextmanager
def tx(con: sqlite3.Connection):
    """BEGIN IMMEDIATE transaction (writers serialize; readers never block)."""
    for attempt in range(20):
        try:
            con.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError:
            time.sleep(0.25 * (attempt + 1))
    else:
        raise RuntimeError("db busy")
    try:
        yield con
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def event(con: sqlite3.Connection, kind: str, detail: str = "") -> None:
    con.execute("INSERT INTO events(ts, kind, detail) VALUES (?,?,?)", (time.time(), kind, str(detail)[:500]))


def kv_get(con: sqlite3.Connection, k: str, default: str = "") -> str:
    r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r["v"] if r else default


def kv_set(con: sqlite3.Connection, k: str, v: str) -> None:
    con.execute("INSERT INTO kv(k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))


def set_state(con: sqlite3.Connection, item_id: str, state: str, **fields) -> None:
    fields["state"] = state
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE items SET {cols} WHERE id=?", (*fields.values(), item_id))


# ---------------------------------------------------------------- naming / dedupe
def slug(s: str, n: int = 40) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:n].rstrip("_") or "video"


def title_fingerprint(title: str) -> str:
    """Dedupe key for the same video uploaded twice (Commons mirror of a YouTube upload, re-uploads):
    lower-cased alphanumerics of the whole title (crawl adds a duration bucket)."""
    t = unicodedata.normalize("NFKD", title or "").lower()
    t = re.sub(r"[^a-z0-9\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", t)
    return t[:200]


def item_slug(item_id: str, title: str) -> str:
    h = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:6]
    return f"{slug(title, 36)}_{h}"


def chunk_plan(duration_s: float) -> List[Tuple[float, float]]:
    """(start, length) per chunk: equal chunks of <= CHUNK_S, so a 25-min video is 3 x 500 s, not
    2 x 600 s + a 300 s tail; anything <= CHUNK_S is one chunk."""
    if duration_s <= CHUNK_S:
        return [(0.0, duration_s)]
    n = int(math.ceil(duration_s / CHUNK_S))
    L = duration_s / n
    return [(i * L, L) for i in range(n)]


# ---------------------------------------------------------------- language guess
SCRIPT_RANGES = [
    ("ru", "\u0400\u04ff"), ("ar", "\u0600\u06ff"), ("fa", "\u0750\u077f"), ("he", "\u0590\u05ff"),
    ("hi", "\u0900\u097f"), ("bn", "\u0980\u09ff"), ("ta", "\u0b80\u0bff"), ("te", "\u0c00\u0c7f"),
    ("th", "\u0e00\u0e7f"), ("ko", "\uac00\ud7af"), ("ja", "\u3040\u30ff"), ("zh", "\u4e00\u9fff"),
    ("el", "\u0370\u03ff"), ("hy", "\u0530\u058f"), ("ka", "\u10a0\u10ff"), ("my", "\u1000\u109f"),
    ("km", "\u1780\u17ff"), ("am", "\u1200\u137f"),
]


def guess_language(title: str, info: Optional[dict] = None, query_lang: str = "") -> Tuple[str, str]:
    """(lang, evidence). yt-dlp's ``language`` field, else the '-orig' automatic-caption track
    (YouTube marks the spoken language that way), else the title's script, else the query's language."""
    info = info or {}
    if info.get("language"):
        return str(info["language"])[:5], "info-json language"
    ac = info.get("automatic_captions") or {}
    orig = [k for k in ac if str(k).endswith("-orig")]
    if orig:
        return orig[0].split("-")[0], "automatic_captions *-orig track"
    counts: Dict[str, int] = {}
    for ch in title or "":
        for lang, rng in SCRIPT_RANGES:
            if rng[0] <= ch <= rng[1]:
                counts[lang] = counts.get(lang, 0) + 1
    if counts:
        best = max(counts, key=counts.get)
        if best == "zh" and counts.get("ja"):
            best = "ja"
        return best, "title script"
    if query_lang:
        return query_lang, "query language"
    if re.search(r"[A-Za-z]", title or ""):
        return "und-latin", "title script (Latin, language unknown)"
    return "und", "unknown"


# ---------------------------------------------------------------- speaker / series
def speaker_key(backend: str, channel_key: str, artist: str, title: str, name: str) -> str:
    """Who is on screen. YouTube: the channel (a vlog channel is one person; an interview channel
    over-merges, which is the safe direction for a per-speaker cap). Commons/archive: the same
    rules as scripts/data_capture_batch.speaker_key."""
    if backend == "ytdlp" and channel_key:
        return "yt:" + channel_key
    try:
        from data_capture_batch import speaker_key as sk
        return sk({"title": title, "artist": artist, "name": name})
    except Exception:
        return slug(artist or name, 40)


def series_key(title: str, query: str, source_kind: str) -> str:
    t = f"{title} {query}".lower()
    if source_kind == "usgov":
        if "hearing" in t or "markup" in t:
            return "hearing"
        if "briefing" in t or "press" in t:
            return "briefing"
        if "address" in t or "remarks" in t:
            return "address"
        return "usgov_other"
    for k, s in (("vlog", "vlog"), ("day in", "vlog"), ("storytime", "vlog"), ("talking to camera", "vlog"),
                 ("podcast", "podcast"), ("lecture", "lecture"), ("lesson", "lecture"), ("tutorial", "lecture"),
                 ("interview", "interview"), ("entrevista", "interview"), ("intervju", "interview"),
                 ("q&a", "qa"), ("q and a", "qa"), ("review", "review"), ("testimony", "testimony"),
                 ("oral history", "oral_history"), ("sermon", "talk"), ("speech", "talk"), ("talk", "talk")):
        if k in t:
            return s
    return "other"


# ---------------------------------------------------------------- video tools
def ffprobe(path: str) -> Dict:
    try:
        out = subprocess.run([bin_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=120)
        j = json.loads(out.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return {"width": s.get("width"), "height": s.get("height"), "fps": round(fps, 3),
                "duration_s": round(float(j.get("format", {}).get("duration", 0) or 0), 2)}
    except Exception:
        return {}


def split_chunk(src: str, dst: str, start: float, length: float) -> bool:
    """Stream-copy one chunk. ``-ss`` before ``-i`` seeks to the keyframe at or before ``start``, so a
    chunk may begin up to one GOP (<= ~5 s) early and overlap the previous chunk's tail; a copy cut
    cannot do better and a re-encode would cost ~10 % of the capture budget."""
    cmd = [bin_path("ffmpeg"), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src, "-t", f"{length:.3f}",
           "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 100_000


def wav_to_opus(wav: str, opus: str) -> bool:
    cmd = [bin_path("ffmpeg"), "-v", "error", "-y", "-i", wav, "-c:a", "libopus", "-b:a", f"{OPUS_KBPS}k",
           "-vbr", "on", "-application", "audio", opus]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return r.returncode == 0 and os.path.exists(opus) and os.path.getsize(opus) > 1000


_tls = threading.local()   # one YuNet per worker thread (FaceDetectorYN is not thread-safe)


def prescreen(path: str, n_frames: int = PRESCREEN_FRAMES) -> Dict:
    """Cheap face check before spending ~1x-realtime capture on a chunk: sample ``n_frames`` evenly,
    run YuNet (the same detector capture's small-face fallback uses), count frames with exactly one
    face. Returns {frac_one_face, frac_multi, frac_none, n, ok}."""
    import cv2
    from animacy.capture import YUNET_SCORE, ensure_model

    yunet = getattr(_tls, "yunet", None)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "n": 0, "error": "cannot open"}
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    dur = n_total / fps if n_total > 0 and fps > 0 else 0.0
    one = multi = none = 0
    sizes: List[float] = []
    for i in range(n_frames):
        t = (i + 0.5) / n_frames * dur if dur > 0 else i * 10.0
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if yunet is None:
            yunet = cv2.FaceDetectorYN.create(ensure_model("face_detection_yunet_2023mar.onnx"), "", (w, h),
                                              score_threshold=YUNET_SCORE)
            _tls.yunet = yunet
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(frame)
        k = 0 if faces is None else len(faces)
        if k == 0:
            none += 1
            continue
        widths = sorted((float(f[2]) for f in faces), reverse=True)
        # one clearly dominant face counts as single: posters / thumbnails / a background second person
        if k == 1 or widths[1] < 0.5 * widths[0]:
            one += 1
            sizes.append(widths[0] / w)
        else:
            multi += 1
    cap.release()
    n = one + multi + none
    frac = one / n if n else 0.0
    return {"ok": n > 0 and frac >= PRESCREEN_MIN_FRAC, "n": n, "frac_one_face": round(frac, 3),
            "frac_multi": round(multi / n, 3) if n else 0.0, "frac_none": round(none / n, 3) if n else 0.0,
            "median_face_w": round(sorted(sizes)[len(sizes) // 2], 3) if sizes else 0.0}


# ---------------------------------------------------------------- misc
def hours(sec: Optional[float]) -> float:
    return (sec or 0.0) / 3600.0


def load_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
