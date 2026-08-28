"""Shared plumbing for the animacy demo-film edit.

Owns: paths, ffmpeg/ffprobe discovery, script parsing, manifest ingestion,
the cut plan, and the EDL (edit decision list) that both the MLT project and
the ffmpeg assembly are generated from.

Nothing here renders. edit_assets.py makes the generated media, edit_mlt.py
writes the Kdenlive project, edit_render.py produces the deliverables.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
SCRIPT_MD = REPO / "docs" / "video" / "script.md"

DATA_VIDEO = REPO / "data" / "video"
VOICE_DIR = DATA_VIDEO / "voice"
PODCAST_DIR = DATA_VIDEO / "podcast"
BROLL_DIR = DATA_VIDEO / "broll"

EDIT_DATA = DATA_VIDEO / "edit"          # generated intermediates (gitignored-ish)
EDIT_DOCS = REPO / "docs" / "video" / "edit"  # the project file + README
MEDIA = REPO / "docs" / "media"          # deliverables

GEN = EDIT_DATA / "gen"                  # generated media: slugs, titles, tone
SHOTS = EDIT_DATA / "shots"              # normalised per-shot intermediates
QC = EDIT_DATA / "qc"                    # extracted frames we look at

EDL_JSON = EDIT_DATA / "edl.json"
BUILD_LOG = EDIT_DATA / "build_log.txt"

FPS = 30
W, H = 1920, 1080

CAMERAS = ["A", "B", "C", "D", "E"]
CAM_NAME = {
    "A": "wide two-shot",
    "B": "lamp single",
    "C": "reachy single",
    "D": "over-shoulder",
    "E": "slow push-in",
}

# --------------------------------------------------------------------------
# tool discovery
# --------------------------------------------------------------------------


def _find(name: str) -> str | None:
    hit = shutil.which(name)
    if hit:
        return hit
    # winget's ffmpeg lands here and is not always on PATH for child shells
    for cand in Path.home().glob(
        f"AppData/Local/Microsoft/WinGet/Packages/*FFmpeg*/**/bin/{name}.exe"
    ):
        return str(cand)
    return None


FFMPEG = _find("ffmpeg")
FFPROBE = _find("ffprobe")


def find_melt() -> str | None:
    """melt.exe, from PATH or a Kdenlive install we may have made."""
    hit = shutil.which("melt") or shutil.which("melt.exe")
    if hit:
        return hit
    direct = [
        Path("C:/Users/sarta/mlt-portable/Shotcut/melt.exe"),
        Path("C:/Program Files/Shotcut/melt.exe"),
    ]
    for c in direct:
        if c.exists():
            return str(c)
    roots = [
        Path("C:/Users/sarta/mlt-portable"),
        Path("C:/Users/sarta/kdenlive-portable"),
        Path("C:/Program Files/kdenlive"),
        Path("C:/Program Files/Kdenlive"),
        Path(r"C:/Program Files (x86)/kdenlive"),
    ]
    for r in roots:
        if not r.exists():
            continue
        for cand in r.glob("**/melt.exe"):
            return str(cand)
    return None


def run(cmd: list[str], *, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 and not quiet:
        sys.stderr.write(f"\n$ {' '.join(str(c) for c in cmd[:6])} ...\n")
        sys.stderr.write((proc.stderr or "")[-4000:] + "\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd[0]}")
    return proc


def ff(args: list[str], *, check: bool = True, quiet: bool = True) -> subprocess.CompletedProcess:
    if not FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH")
    return run([FFMPEG, "-hide_banner", "-nostdin", "-y", *args], check=check, quiet=quiet)


_probe_cache: dict[str, dict] = {}


def probe(path: Path | str) -> dict:
    """{'dur': float, 'has_video': bool, 'has_audio': bool, 'w': int, 'h': int}"""
    key = str(path)
    if key in _probe_cache:
        return _probe_cache[key]
    out = {"dur": 0.0, "has_video": False, "has_audio": False, "w": 0, "h": 0}
    if FFPROBE and Path(path).exists():
        p = run(
            [FFPROBE, "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", key],
            check=False, quiet=True,
        )
        if p.returncode == 0:
            try:
                j = json.loads(p.stdout)
                out["dur"] = float(j.get("format", {}).get("duration") or 0.0)
                for st in j.get("streams", []):
                    if st.get("codec_type") == "video":
                        out["has_video"] = True
                        out["w"] = int(st.get("width") or 0)
                        out["h"] = int(st.get("height") or 0)
                        if not out["dur"]:
                            out["dur"] = float(st.get("duration") or 0.0)
                    elif st.get("codec_type") == "audio":
                        out["has_audio"] = True
                        if not out["dur"]:
                            out["dur"] = float(st.get("duration") or 0.0)
            except Exception:
                pass
    _probe_cache[key] = out
    return out


def frames(seconds: float) -> int:
    return int(round(seconds * FPS))


def secs(frame_count: int) -> float:
    return frame_count / FPS


def tc(seconds: float) -> str:
    """HH:MM:SS.mmm, the timecode form MLT accepts."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


# --------------------------------------------------------------------------
# the script
# --------------------------------------------------------------------------


@dataclass
class Line:
    idx: int                 # global order, 0-based
    section: int
    section_title: str
    section_over: str
    speaker: str             # LAMP | REACHY
    text: str
    sec_pos: int = 0         # position within its section AS WRITTEN; the cut
                             # plan is keyed on this so dropping a line for
                             # length does not slide every later shot onto the
                             # wrong camera
    wav: str | None = None   # source narration file, if the voice agent shipped one
    dur: float = 0.0
    est: bool = True         # duration is an estimate, not measured
    start: float = 0.0       # timeline position, filled by build_edl
    end: float = 0.0
    src_t: float = -1.0      # this line's start in the podcast render timeline
    sec_src_t0: float = -1.0 # its section clip's start in that same timeline
    gap_after: float = -1.0  # the gap the podcast render actually baked in


_SEC_RE = re.compile(r"^##\s*(\d+)\s*[\u2014\u2013-]\s*(.+?)\s*$")
_LINE_RE = re.compile(r"^\*\*(LAMP|REACHY)\s*:?\s*\*\*\s*:?\s*(.+?)\s*$")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def estimate_duration(text: str) -> float:
    """Rough spoken length. Deliberately generous: these two talk deliberately."""
    words = len(re.findall(r"[\w'\u2019-]+", text))
    return round(words / 2.55 + 0.45, 3)


def parse_script(path: Path = SCRIPT_MD) -> list[Line]:
    if not path.exists():
        raise SystemExit(f"script not found: {path}")
    lines: list[Line] = []
    sec, title, over = 0, "", ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _SEC_RE.match(raw.strip())
        if m:
            sec = int(m.group(1))
            rest = m.group(2)
            om = re.search(r"\(over:\s*(.+?)\)\s*$", rest)
            over = om.group(1).strip() if om else ""
            title = re.sub(r"\s*\(over:.*$", "", rest).strip()
            continue
        m = _LINE_RE.match(raw.strip())
        if m and sec:
            txt = m.group(2).strip()
            lines.append(
                Line(
                    idx=len(lines), section=sec, section_title=title, section_over=over,
                    speaker=m.group(1), text=txt, dur=estimate_duration(txt),
                    sec_pos=sum(1 for l in lines if l.section == sec),
                )
            )
    if not lines:
        raise SystemExit(f"no dialogue lines parsed out of {path}")
    return lines


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _entries(blob) -> list[dict]:
    """Manifests in this repo come in several shapes. Flatten them all."""
    if blob is None:
        return []
    if isinstance(blob, list):
        return [e for e in blob if isinstance(e, dict)]
    if isinstance(blob, dict):
        for key in ("lines", "clips", "items", "entries", "files", "shots", "renders", "broll"):
            v = blob.get(key)
            if isinstance(v, list):
                return [e for e in v if isinstance(e, dict)]
            if isinstance(v, dict):
                out = []
                for k, e in v.items():
                    if isinstance(e, dict):
                        e = dict(e)
                        e.setdefault("id", k)
                        out.append(e)
                return out
        # a bare mapping of id -> record
        if all(isinstance(v, dict) for v in blob.values()) and blob:
            out = []
            for k, e in blob.items():
                e = dict(e)
                e.setdefault("id", k)
                out.append(e)
            return out
    return []


def _pick(entry: dict, *names, default=None):
    for n in names:
        if n in entry and entry[n] not in (None, ""):
            return entry[n]
    return default


SHOW_JSON = PODCAST_DIR / "show.json"


def attach_show(lines: list[Line], log: list[str]) -> bool:
    """Adopt the podcast agent's timeline.

    The host robots' motion is generated from the narration on that timeline
    and baked into the camera renders, so the edit has to speak the same clock.
    Every line picks up its source offset here; the picture is then always
    pulled from the frames that belong to that line, whatever we do with the
    surrounding gaps.
    """
    blob = _load_json(SHOW_JSON)
    if not isinstance(blob, dict) or not blob.get("lines"):
        return False

    sec_t0: dict[int, float] = {}
    for s in blob.get("sections", []):
        n = _sec_num(s.get("number", s.get("index")))
        if n is not None and s.get("t_start") is not None:
            sec_t0[n] = float(s["t_start"])

    entries = blob["lines"]
    by_text = {}
    for e in entries:
        if e.get("text"):
            by_text.setdefault(_norm(str(e["text"])), e)

    matched = 0
    for ln in lines:
        e = by_text.get(_norm(ln.text))
        if e is None:
            continue
        matched += 1
        ln.src_t = float(e.get("t_start", -1.0))
        ln.sec_src_t0 = sec_t0.get(ln.section, -1.0)
        d = e.get("audio_seconds") or e.get("seconds")
        if d:
            ln.dur, ln.est = round(float(d), 3), False
        raw = e.get("wav")
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = PODCAST_DIR / p
            if p.exists():
                ln.wav = str(p)

    # the gap the render actually baked in after each line
    order = sorted([l for l in lines if l.src_t >= 0], key=lambda l: l.src_t)
    for a, b in zip(order, order[1:]):
        a.gap_after = round(max(0.0, b.src_t - (a.src_t + a.dur)), 3)

    placeholder = bool(blob.get("placeholder_voice"))
    log.append(
        f"show.json: adopted the podcast timeline for {matched}/{len(lines)} lines "
        f"({blob.get('seconds')}s as rendered, "
        f"{'PLACEHOLDER' if placeholder else 'final'} voice)"
    )
    if matched < len(lines):
        log.append(f"show.json: WARNING {len(lines) - matched} script line(s) "
                   f"are not in the render - they will use estimates and may desync")
    return matched > 0


def attach_voice(lines: list[Line], log: list[str]) -> None:
    """Bind per-line narration WAVs onto the parsed script lines.

    Matches by normalised text first (order-independent, survives the voice
    agent reordering or renumbering), then falls back to section+order.
    """
    man = _load_json(VOICE_DIR / "manifest.json")
    entries = _entries(man)
    if not entries:
        if any(l.wav for l in lines):
            return   # show.json already supplied audio
        log.append("voice: no manifest.json - every line uses an estimated duration")
        return
    before = {l.idx: (l.dur, l.src_t) for l in lines}

    by_text: dict[str, dict] = {}
    for e in entries:
        t = _pick(e, "text", "line", "utterance", "content", default="")
        if t:
            by_text.setdefault(_norm(str(t)), e)

    used: set[int] = set()
    unmatched: list[Line] = []
    for ln in lines:
        e = by_text.get(_norm(ln.text))
        if e is not None and id(e) not in used:
            used.add(id(e))
            _bind_voice(ln, e)
        else:
            unmatched.append(ln)

    if unmatched:
        leftovers = [e for e in entries if id(e) not in used]
        # positional fallback, in document order
        for ln, e in zip(unmatched, leftovers):
            _bind_voice(ln, e)
            used.add(id(e))

    got = sum(1 for ln in lines if ln.wav)
    log.append(f"voice: matched {got}/{len(lines)} lines to WAVs from {VOICE_DIR / 'manifest.json'}")
    missing = [ln.idx for ln in lines if not ln.wav]
    if missing:
        log.append(f"voice: MISSING audio for line indices {missing} (estimated durations used)")

    # If the real voice is a different length from what the host renders were
    # generated against, the robots' mouths no longer match the words.
    drift = [ln for ln in lines
             if before.get(ln.idx, (0, -1))[1] >= 0
             and abs(ln.dur - before[ln.idx][0]) > 0.15]
    if drift:
        worst = max(abs(ln.dur - before[ln.idx][0]) for ln in drift)
        log.append(
            f"voice: WARNING {len(drift)} line(s) differ from the podcast render "
            f"by up to {worst:.2f}s - the host footage was generated against "
            f"show.json and will drift; ask the podcast agent to re-render"
        )


def _bind_voice(ln: Line, e: dict) -> None:
    raw = _pick(e, "wav", "file", "path", "audio", "filename", "output")
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            for base in (VOICE_DIR, DATA_VIDEO, REPO):
                if (base / p).exists():
                    p = base / p
                    break
        if p.exists():
            ln.wav = str(p)
            d = probe(p)["dur"]
            if d > 0.05:
                ln.dur, ln.est = round(d, 3), False
                return
    d = _pick(e, "duration", "dur", "seconds", "length")
    if isinstance(d, (int, float)) and d > 0.05:
        ln.dur, ln.est = round(float(d), 3), False


FULL_STEMS = {"full", "all", "master", "whole", "complete", "timeline"}

# Host clips do not always carry a section number - "open.mp4", "close.mp4".
# These are the words the script's own section titles give us.
SECTION_WORDS: dict[str, int] = {
    "open": 1, "cold": 1, "coldopen": 1,
    "capture": 2,
    "onefile": 3, "robotmd": 3, "perrobot": 3,
    "retarget": 4,
    "interaction": 5, "layer": 5,
    "hardware": 6, "realhardware": 6,
    "data": 7,
    "judge": 8, "grading": 8,
    "close": 9, "closing": 9,
}


def _infer_section_and_base(p: Path, sec_t0: dict[int, float],
                            sec_dur: dict[int, float]) -> tuple[int, float] | None:
    """Work out which section a host clip covers, and what podcast-timeline
    time its first frame is.

    A clip may be exactly its section, or carry a lead-in or a tail handle.
    Getting the base wrong shifts every cut in that section, so prefer an
    explicit section number, then a word from the section title, then - only as
    a last resort - a duration match.
    """
    stem = _norm(p.stem)
    sec = None
    m = re.search(r"s(?:ec(?:tion)?)?[_\-]?(\d{1,2})", p.stem, re.I)
    if m and int(m.group(1)) in sec_t0:
        sec = int(m.group(1))
    if sec is None:
        for word, n in SECTION_WORDS.items():
            if word in stem and n in sec_t0:
                sec = n
                break
    dur = probe(p)["dur"]
    if sec is None and dur > 0:
        near = [(abs(dur - d), n) for n, d in sec_dur.items() if abs(dur - d) < 2.2]
        if near:
            sec = min(near)[1]
    if sec is None:
        return None

    t0, sd = sec_t0[sec], sec_dur.get(sec, dur)
    extra = dur - sd
    # more material than the section itself: for the opening section the extra
    # is the film's lead-in and sits in front, everywhere else it is tail
    base = t0 - min(extra, t0) if (extra > 0.3 and sec == min(sec_t0)) else t0
    return sec, round(max(0.0, base), 3)


def load_podcast(log: list[str], sec_t0: dict[int, float], show_seconds: float
                 ) -> dict[tuple[str, int], tuple[Path, float]]:
    """(camera, section) -> (clip, clip_start_in_podcast_timeline).

    The podcast agent may ship one clip per section or one long clip per
    camera; both are fine, but the edit has to know what podcast-timeline time
    each clip's first frame is, or every cut lands on the wrong frame. That
    offset is what the second element of the value carries.
    """
    out: dict[tuple[str, int], tuple[Path, float]] = {}
    if not PODCAST_DIR.exists():
        log.append(f"podcast: {PODCAST_DIR} does not exist yet - all host shots are slugs")
        return out

    sections = sorted(sec_t0) or list(range(1, 10))
    full_hits: list[str] = []
    inferred: list[str] = []
    unplaced: list[str] = []

    def note(cam: str, sec: int, p: Path, base: float) -> bool:
        if (cam, sec) not in out and p.exists() and probe(p)["dur"] > 0.3:
            out[(cam, sec)] = (p, base)
            return True
        return False

    def add_full(cam: str, p: Path, base: float = 0.0):
        before = len(out)
        for s in sections:
            note(cam, s, p, base)
        if len(out) > before:
            full_hits.append(f"{cam}:{p.name}")

    def is_full(p: Path) -> bool:
        if p.stem.lower() in FULL_STEMS:
            return True
        d = probe(p)["dur"]
        return bool(show_seconds and d >= 0.85 * show_seconds)

    # 1. an explicit manifest wins - it can state the offset outright
    for e in _entries(_load_json(PODCAST_DIR / "render_manifest.json")):
        raw = _pick(e, "file", "path", "clip", "output", "filename")
        if not raw:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            for b in (PODCAST_DIR, DATA_VIDEO, REPO):
                if (b / p).exists():
                    p = b / p
                    break
        if not p.exists():
            continue
        cam = str(_pick(e, "camera", "cam", "angle", default="") or "").strip().upper()[:1]
        if cam not in CAMERAS:
            cam = _cam_from_path(p)
        if cam not in CAMERAS:
            continue
        stated = _pick(e, "t_start", "clip_start", "start", "offset")
        sec_raw = _pick(e, "section", "sec", "scene", "part")
        if str(sec_raw).strip().lower() in ("all", "*", "full") or is_full(p):
            add_full(cam, p, float(stated) if isinstance(stated, (int, float)) else 0.0)
            continue
        secn = _sec_num(sec_raw if sec_raw is not None else p.stem)
        if secn and secn in sec_t0:
            base = float(stated) if isinstance(stated, (int, float)) else sec_t0[secn]
            note(cam, secn, p, base)

    # 2. otherwise read the layout off disk
    sec_dur = {}
    for s in (_load_json(SHOW_JSON) or {}).get("sections", []):
        n = _sec_num(s.get("number", s.get("index")))
        if n is not None and s.get("t_start") is not None and s.get("t_end") is not None:
            sec_dur[n] = float(s["t_end"]) - float(s["t_start"])

    candidates = [p for cam in CAMERAS for p in
                  (sorted((PODCAST_DIR / cam).glob("*.mp4"))
                   if (PODCAST_DIR / cam).exists() else [])]
    candidates += sorted(PODCAST_DIR.glob("*.mp4"))
    for p in candidates:
        cam = _cam_from_path(p)
        if cam not in CAMERAS:
            continue
        if is_full(p):
            add_full(cam, p)
            continue
        got = _infer_section_and_base(p, sec_t0, sec_dur)
        if got:
            secn, base = got
            if note(cam, secn, p, base):
                inferred.append(f"{cam}/{p.name}->s{secn}@{base:.2f}s")
        elif probe(p)["dur"] > 0.3:
            unplaced.append(f"{cam}/{p.name}")

    cams = sorted({c for c, _ in out})
    log.append(f"podcast: {len(out)} (camera, section) slot(s) covered, cameras {cams or 'none'}"
               + (f"; full-timeline: {full_hits}" if full_hits else ""))
    if inferred:
        log.append(f"podcast: {len(inferred)} clip(s) placed by inference rather than "
                   f"by render_manifest.json: {inferred}")
    if unplaced:
        log.append(f"podcast: WARNING could not place {unplaced} - "
                   f"no section number, no title word, no duration match")
    return out


def _sec_num(value) -> int | None:
    if isinstance(value, int):
        return value if 1 <= value <= 99 else None
    m = re.search(r"(\d{1,2})", str(value))
    if m:
        n = int(m.group(1))
        if 1 <= n <= 99:
            return n
    return None


def _cam_from_path(p: Path) -> str:
    if p.parent.name.upper() in CAMERAS:
        return p.parent.name.upper()
    m = re.match(r"^([A-E])[_\-. ]", p.stem.upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:cam|camera)[_\-]?([A-E])\b", p.stem, re.I)
    return m.group(1).upper() if m else ""


BROLL_KEYWORDS: dict[str, list[str]] = {
    "tracker_overlay": ["tracker", "overlay", "landmark", "face", "mesh", "talkinghead",
                        "talking", "webcam", "capture", "pose", "mediapipe"],
    "channels": ["channel", "canonical", "bars", "signal", "28", "twentyeight", "trace", "strip"],
    "robot_md": ["robotmd", "robot_md", "markdown", "scroll", "schema", "licence",
                 "license", "restpose", "joints", "spec"],
    "check_pass": ["check", "terminal", "cli", "console", "pass", "validate", "green", "shell"],
    "vendor_nod": ["vendornod", "vendor", "ab", "nod", "handmade", "compare", "sidebyside"],
    "viewer_ab": ["leanin", "lean", "gaze", "viewer", "forward", "kinematics", "stays"],
    "speed_cap": ["speed", "cap", "limit", "clearance", "violation", "plot", "chart", "ceiling"],
    "waveform": ["wave", "audio", "sync", "interaction", "talk", "spectro", "clock"],
    "hardware": ["desk", "physical", "hardware", "reachy", "mini", "benchtop"],
    "readback": ["readback", "read_back", "sim2real", "daemon", "measured",
                 "commanded", "axis", "degrees", "evidence"],
    "csv_validator": ["csv", "autonomous", "validator", "export", "os", "column", "lerobot"],
    "mapping": ["mapping", "gains", "fitted", "baseyaw", "headyaw", "torsoyaw", "mixed"],
    "dataset": ["huggingface", "hugging", "datasetpage", "dataset", "card", "licence",
                "license"],
    "data_report": ["datareport", "report", "speakers", "minutes", "kept", "totals",
                    "harvest", "counter", "breakdown"],
    "grading_reel": ["reel", "blind", "clip9", "numbered", "watched", "judge"],
    "score_table": ["scoretable", "score", "table", "published", "rank", "overall"],
}

# Words that mean "this is the other clip in the same section". Without these,
# two section-3 clips that both mention ROBOT.md score identically and the
# assignment lands inverted - the terminal under "your robot is one markdown
# file" and the file scroll under "animacy check passing".
BROLL_NEGATIVE: dict[str, list[str]] = {
    "robot_md": ["check", "validating", "validate"],
    "check_pass": ["scroll", "schema"],
    "vendor_nod": ["leanin", "gaze", "csv"],
    "viewer_ab": ["vendornod", "handmade", "csv"],
    "hardware": ["evidence", "table", "page"],
    "readback": ["desk"],
    # section 5 is "what we are doing right now" - replaying the section 2
    # capture footage under it reads as a repeat. Better to stay on the hosts.
    "waveform": ["capture", "tracker", "preview", "landmark"],
    "mapping": ["ab", "vendornod", "csv", "leanin"],
    "vendor_nod": ["mapping", "gains"],
    "dataset": ["datareport", "speakers", "kept"],
    "data_report": ["huggingface", "hugging"],
    "grading_reel": ["scoretable", "published", "overall"],
    "score_table": ["reel", "blind", "clip9"],
}


@dataclass
class BrollClip:
    path: Path
    dur: float
    haystack: str
    section: int | None = None   # the b-roll manifest tags most clips by section
    portrait: bool = False
    used: float = 0.0     # how far into it we have already spent


class BrollPool:
    """Maps abstract b-roll slots onto whatever clips actually exist.

    The b-roll agent may ship five clips or fifteen, under names we do not
    control, so slots are matched by keyword score. When one clip has to serve
    two slots we advance the in-point instead of replaying the same frames.
    """

    def __init__(self, log: list[str]):
        self.clips: list[BrollClip] = []
        self.log = log
        self.assigned: dict[str, BrollClip] = {}
        if not BROLL_DIR.exists():
            log.append(f"broll: {BROLL_DIR} does not exist yet - all b-roll shots are slugs")
            return

        meta: dict[str, str] = {}
        sections: dict[str, int] = {}
        for e in _entries(_load_json(BROLL_DIR / "manifest.json")):
            raw = _pick(e, "file", "path", "clip", "output", "filename", "id")
            if not raw:
                continue
            stem = Path(str(raw)).stem
            bits = [str(_pick(e, "id", default="")), str(_pick(e, "title", "name", default="")),
                    str(_pick(e, "shows", "description", "desc", "caption", "text", default="")),
                    str(_pick(e, "source", default="")),
                    " ".join(str(t) for t in (_pick(e, "tags", "keywords", default=[]) or []))]
            meta[stem] = " ".join(bits)
            sec = _sec_num(_pick(e, "section", "sec", "scene"))
            if sec:
                sections[stem] = sec

        for p in sorted(BROLL_DIR.glob("**/*.mp4")):
            info = probe(p)
            if info["dur"] < 0.4:
                continue
            hay = _norm(p.stem + " " + p.parent.name + " " + meta.get(p.stem, ""))
            sec = sections.get(p.stem) or _sec_num(re.match(r"^s(\d+)", p.stem).group(1)
                                                   if re.match(r"^s(\d+)", p.stem) else None)
            self.clips.append(BrollClip(
                path=p, dur=info["dur"], haystack=hay, section=sec,
                portrait=bool(info["h"] and info["w"] and info["w"] < info["h"]),
            ))
        tagged = sum(1 for c in self.clips if c.section)
        log.append(f"broll: found {len(self.clips)} clips in {BROLL_DIR} "
                   f"({tagged} tagged with a section)")

    def _score(self, slot: str, section: int | None, need: float, c: BrollClip) -> float:
        words = BROLL_KEYWORDS.get(slot, [slot])
        score = sum(2.0 if w in c.haystack else 0.0 for w in words)
        score -= sum(2.5 if w in c.haystack else 0.0
                     for w in BROLL_NEGATIVE.get(slot, []))
        if section and c.section == section:
            score += 3.0              # the b-roll agent's own section tag
        elif section and c.section:
            score -= 1.5              # shot for a different part of the film
        if c.dur >= need:
            score += 0.5
        if c.portrait:
            score -= 2.5              # would pillarbox into a 16:9 frame
        return score

    MIN_SCORE = 1.5   # below this we would rather stay on the hosts

    def plan(self, requests: list[tuple[str, int, float]], log: list[str]) -> None:
        """Assign clips to slots globally rather than first-come-first-served.

        Taking the best match for each slot in script order strands good
        footage: two section-3 clips both mention ROBOT.md, the first slot grabs
        whichever sorts first, and the actual slow scroll over the file never
        makes the cut. Matching the highest-scoring pairs first, one clip to one
        slot, uses everything that was shot.
        """
        if not self.clips:
            return
        seen: dict[str, tuple[str, int, float]] = {}
        for slot, sec, need in requests:
            prev = seen.get(slot)
            if prev is None or need > prev[2]:
                seen[slot] = (slot, sec, need)
        slots = list(seen.values())

        scores = {(slot, id(c)): self._score(slot, sec, need, c)
                  for slot, sec, need in slots for c in self.clips}
        pairs = sorted(
            ((scores[(slot, id(c))], slot, c) for slot, _s, _n in slots for c in self.clips),
            key=lambda x: -x[0],
        )
        open_slots = {s for s, _, _ in slots}
        free = {id(c) for c in self.clips}
        for sc, slot, c in pairs:
            if slot in open_slots and id(c) in free and sc >= self.MIN_SCORE:
                self.assigned[slot] = c
                open_slots.discard(slot)
                free.discard(id(c))

        # slots with nothing distinct left may share a clip, but only if the
        # match is real; otherwise they stay unassigned and fall back to the wide
        for slot, sec, need in slots:
            if slot not in open_slots:
                continue
            best = max(self.clips, key=lambda c: scores[(slot, id(c))])
            if scores[(slot, id(best))] >= self.MIN_SCORE:
                self.assigned[slot] = best
                open_slots.discard(slot)

        unused = [c.path.name for c in self.clips if id(c) in free
                  and c not in self.assigned.values()]
        if open_slots:
            log.append(f"broll: no usable footage for {sorted(open_slots)} - "
                       f"those shots stay on the hosts")
        if unused:
            log.append(f"broll: delivered but unused: {sorted(unused)}")

    def take(self, slot: str, need: float, section: int | None = None
             ) -> tuple[Path, float] | None:
        """Return (path, in_point) for this slot, or None if we have nothing."""
        clip = self.assigned.get(slot)
        if clip is None:
            return None

        start = clip.used
        if start + need > clip.dur:
            start = 0.0 if need >= clip.dur else max(0.0, min(start, clip.dur - need))
            if clip.used > 0 and clip.dur - need > 0.05:
                start = 0.0
        clip.used = min(clip.dur, start + need)
        return clip.path, round(start, 3)


# --------------------------------------------------------------------------
# the cut plan
# --------------------------------------------------------------------------
#
# One entry per spoken line, in script order. Each entry is the list of shots
# that line is cut across: one shot means the line plays on a single angle, two
# means there is an internal cut roughly 55% of the way through (only taken if
# the line is long enough to earn it). "A".."E" are host cameras; "b:<slot>"
# pulls from the b-roll pool.
#
# The shape of it: wide for exchanges, singles for single lines, b-roll while
# they explain a step, back to the wide for the punchline that closes a section.

CUTPLAN: dict[int, list[list[str]]] = {
    # The cold open plays on the slow push-in: it is the one shot that says
    # "sit down, this is a conversation" without a title card doing it.
    1: [["E"], ["C"], ["B"], ["A"], ["B"]],
    2: [["b:tracker_overlay"], ["b:channels", "B"], ["b:channels"], ["A"]],
    3: [["C"], ["b:robot_md"], ["D"], ["b:check_pass", "A"]],
    4: [["b:mapping", "b:vendor_nod"], ["b:viewer_ab"], ["b:speed_cap", "A"]],
    5: [["A"], ["B"], ["C"], ["b:waveform"], ["C"], ["A"]],
    6: [["b:hardware", "b:readback"], ["b:csv_validator"], ["A"], ["B"]],
    7: [["b:dataset"], ["b:data_report"], ["A"]],
    8: [["B", "b:grading_reel"], ["b:grading_reel"], ["b:score_table", "A"]],
    9: [["A"], ["E"]],
}

# Section boundaries that get a dissolve rather than a hard cut, keyed by the
# incoming section.
#
# All five cameras look at the same two robots on the same locked-off set, so
# dissolving one host angle into another double-exposes the robots and reads as
# a glitch rather than a transition. Every dissolve here therefore lands on a
# boundary where the picture changes character - a host shot into b-roll - and
# build_edl refuses any dissolve that would end up host-to-host anyway.
# Spaced across the film at roughly 16 s, 72 s and 140 s.
DISSOLVES: dict[int, float] = {2: 0.60, 4: 0.55, 6: 0.65}

# Section boundaries where the narration leads its picture (a J-cut).
JCUTS: dict[int, float] = {3: 0.55, 7: 0.50, 8: 0.60}

# At most five, small, bottom-left, one line, held ~2.5 s on the section's cut.
LOWER_THIRDS: dict[int, str] = {
    2: "capture",
    3: "one file per robot",
    4: "retarget",
    5: "the interaction layer",
    6: "on real hardware",
}
LT_HOLD = 2.5
LT_FADE = 6 / FPS

GAP_WITHIN = 0.14       # between lines inside a section
GAP_SECTION = 0.46      # the breath at a section boundary
SECTION_GAP_CAP = 0.55  # the most we let a rendered section gap run to
GAP_AFTER_PUNCH = 0.30  # extra beat after a deadpan one-liner
LEAD_IN = 0.35          # picture before the first word
END_BEAT = 0.70         # stillness after the last word, before the end card
ENDCARD_HOLD = 3.0
TAIL_BLACK = 0.5

# Lines that may be dropped, worst-first, if the cut runs past the length
# ceiling. Identified as (section, sec_pos) against the script as written.
# Ordered to protect the two things that make this film credible: the
# SO-101 proof point, and the admission that the learned model loses on beat
# alignment. Those go last, and the learned-model pair goes together.
DROP_ORDER: list[tuple[int, int]] = [
    (2, 2),           # "that is the canonical space" - restates the line before it
    (1, 2),           # "and when the menu runs out, we repeat ourselves"
    (6, 1),           # the Autonomous OS CSV line - the b-roll already shows it
    (8, 1),           # "it is blind, the test lines are sealed"
    (5, 4), (5, 5),   # the learned-model admission, last resort and as a pair
]
MAX_RUNTIME = 210.0   # 3:30 ceiling for narration+picture, before the end card


# --------------------------------------------------------------------------
# EDL
# --------------------------------------------------------------------------


@dataclass
class Shot:
    start: float
    dur: float
    kind: str                 # "cam" | "broll" | "slug" | "endcard"
    label: str                # "A" / "b:dataset" / "endcard"
    src: str | None = None    # resolved file, None => slug
    src_in: float = 0.0
    section: int = 0
    line_idx: int = -1
    note: str = ""

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class AudioCue:
    start: float
    dur: float
    src: str | None
    line_idx: int
    speaker: str
    text: str


@dataclass
class Title:
    start: float
    dur: float
    text: str
    png: str | None = None


@dataclass
class EDL:
    shots: list[Shot] = field(default_factory=list)
    narration: list[AudioCue] = field(default_factory=list)
    titles: list[Title] = field(default_factory=list)
    dissolves: list[dict] = field(default_factory=list)   # {at, dur, section}
    roomtone: list[dict] = field(default_factory=list)    # {start, dur}
    lines: list[Line] = field(default_factory=list)
    total: float = 0.0
    log: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "fps": FPS, "width": W, "height": H, "total": round(self.total, 3),
            "shots": [asdict(s) for s in self.shots],
            "narration": [asdict(a) for a in self.narration],
            "titles": [asdict(t) for t in self.titles],
            "dissolves": self.dissolves,
            "roomtone": self.roomtone,
            "lines": [asdict(l) for l in self.lines],
            "dropped": self.dropped,
            "log": self.log,
        }


def _cover_angle(prev: str, plan: list[list[str]]) -> str:
    """A different angle from `prev`, preferring one this section already uses."""
    in_plan = [lbl for spec in plan for lbl in spec
               if lbl in CAMERAS and lbl != prev]
    for want in ("A", "D", "B", "C", "E"):
        if want in in_plan:
            return want
    return "A" if prev != "A" else "D"


def build_edl(*, max_runtime: float = MAX_RUNTIME) -> EDL:
    log: list[str] = []
    lines = parse_script()
    attach_show(lines, log)
    attach_voice(lines, log)
    show_blob = _load_json(SHOW_JSON) or {}
    sec_t0 = {n: t for n, t in
              ((_sec_num(s.get("number", s.get("index"))), s.get("t_start"))
               for s in show_blob.get("sections", []))
              if n is not None and isinstance(t, (int, float))}
    podcast = load_podcast(log, sec_t0, float(show_blob.get("seconds") or 0.0))
    broll = BrollPool(log)

    # --- length discipline: drop the marked lines until we fit -------------
    dropped: list[str] = []
    orig_pos = {(l.section, l.sec_pos): i for i, l in enumerate(lines)}
    by_sec: dict[int, list[Line]] = {}
    for ln in lines:
        by_sec.setdefault(ln.section, []).append(ln)

    def lay_out(seq: list[Line]) -> float:
        """Walk the narration exactly as the timeline will, returning the total
        film length including the end card.

        Where the podcast render baked in a gap, we keep it: shortening a gap
        inside a section would slide the picture against the motion that was
        generated for it. The one gap we do tighten is the one at a section
        boundary, because that is always a cut to a different source anyway.
        """
        t = LEAD_IN
        last_end = t
        for i, ln in enumerate(seq):
            ln.start = round(t, 3)
            ln.end = last_end = round(t + ln.dur, 3)
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if nxt is None:
                break
            boundary = nxt.section != ln.section
            if ln.gap_after >= 0:
                gap = min(ln.gap_after, SECTION_GAP_CAP) if boundary else ln.gap_after
            else:
                gap = GAP_SECTION if boundary else GAP_WITHIN
                if len(ln.text.split()) <= 5:
                    gap += GAP_AFTER_PUNCH
            t = ln.end + gap
        return last_end + END_BEAT + ENDCARD_HOLD + TAIL_BLACK

    def projected() -> float:
        return lay_out([ln for s in sorted(by_sec) for ln in by_sec[s]])

    for sec, pos in DROP_ORDER:
        if projected() <= max_runtime:
            break
        pool = by_sec.get(sec, [])
        hit = next((l for l in pool if l.sec_pos == pos), None)
        if hit is not None:
            pool.remove(hit)
            dropped.append(f"s{sec}.{pos} {hit.speaker}: {hit.text[:70]}")
    if dropped:
        log.append(f"length: dropped {len(dropped)} line(s) to fit {mmss(max_runtime)}")
    if projected() > max_runtime:
        log.append(
            f"length: WARNING still {mmss(projected())} after drops - over the "
            f"{mmss(max_runtime)} ceiling; narration needs tightening at the source"
        )

    kept = [ln for sec in sorted(by_sec) for ln in by_sec[sec]]
    for i, ln in enumerate(kept):
        ln.idx = i

    # --- lay the narration down -------------------------------------------
    edl = EDL(lines=kept, log=log, dropped=dropped)
    lay_out(kept)
    narration_end = kept[-1].end
    for ln in kept:
        edl.narration.append(
            AudioCue(start=ln.start, dur=ln.dur, src=ln.wav,
                     line_idx=ln.idx, speaker=ln.speaker, text=ln.text)
        )

    # --- picture ----------------------------------------------------------
    # Each line's picture window runs from its own start to the next line's
    # start, so cuts land on the ends of sentences. Section boundaries are
    # then shifted for J-cuts.
    windows: list[tuple[Line, float, float]] = []
    for i, ln in enumerate(kept):
        w_start = ln.start if i else 0.0          # first shot covers the lead-in
        w_end = kept[i + 1].start if i + 1 < len(kept) else narration_end + END_BEAT
        windows.append((ln, w_start, w_end))

    # J-cut: hold the outgoing picture past the incoming section's first word.
    adj: list[list[float]] = [[ws, we] for _, ws, we in windows]
    for i, (ln, _, _) in enumerate(windows):
        if i == 0:
            continue
        prev = kept[i - 1]
        if ln.section != prev.section and ln.section in JCUTS:
            lead = JCUTS[ln.section]
            new_cut = min(adj[i][0] + lead, adj[i][1] - 0.5)
            if new_cut > adj[i][0]:
                adj[i - 1][1] = new_cut
                adj[i][0] = new_cut
                log.append(f"j-cut: section {ln.section} audio leads picture by "
                           f"{new_cut - ln.start:.2f}s")

    # Expand each window into its planned shots.
    raw_shots: list[Shot] = []
    for (ln, _, _), (ws, we) in zip(windows, adj):
        k = ln.sec_pos
        plan = CUTPLAN.get(ln.section, [["A"]])
        spec = plan[k] if k < len(plan) else plan[-1] if plan else ["A"]
        span = max(0.2, we - ws)
        parts = spec if (len(spec) > 1 and span >= 4.2) else spec[:1]
        if len(parts) == 2:
            split = ws + span * 0.55
            bounds = [(ws, split), (split, we)]
        else:
            bounds = [(ws, we)]
        for label, (a, b) in zip(parts, bounds):
            raw_shots.append(
                Shot(start=round(a, 3), dur=round(b - a, 3), kind="", label=label,
                     section=ln.section, line_idx=ln.idx)
            )

    # Where a line was dropped for length, the host clip either side of the
    # join is one continuous render, so staying on the same angle would read as
    # a jump cut. Cover it the way an editor would: change the angle.
    covered = 0
    for a, b in zip(kept, kept[1:]):
        if orig_pos[(b.section, b.sec_pos)] == orig_pos[(a.section, a.sec_pos)] + 1:
            continue
        before = next((s for s in reversed(raw_shots) if s.line_idx == a.idx), None)
        after = next((s for s in raw_shots if s.line_idx == b.idx), None)
        if before is None or after is None:
            continue
        if before.label == after.label and after.label in CAMERAS:
            after.label = _cover_angle(before.label, CUTPLAN.get(b.section, []))
            covered += 1
    if covered:
        log.append(f"continuity: {covered} join(s) from dropped lines covered "
                   f"by changing the angle")

    # Merge neighbours that call for the same angle: no cut where none is wanted.
    merged: list[Shot] = []
    for s in raw_shots:
        if merged and merged[-1].label == s.label and abs(merged[-1].end - s.start) < 0.02:
            merged[-1].dur = round(s.end - merged[-1].start, 3)
        else:
            merged.append(s)

    # --- resolve every shot to a real file, or a slug ----------------------
    # Decide the whole b-roll assignment first, so no slot strands footage a
    # later slot needed more.
    broll.plan([(s.label[2:], s.section, s.dur)
                for s in merged if s.label.startswith("b:")], log)
    for s in merged:
        if s.label.startswith("b:") and s.label[2:] not in broll.assigned:
            s.label = "A"      # nothing suitable was shot: stay on the wide
    merged2: list[Shot] = []
    for s in merged:
        if merged2 and merged2[-1].label == s.label and abs(merged2[-1].end - s.start) < 0.02:
            merged2[-1].dur = round(s.end - merged2[-1].start, 3)
        else:
            merged2.append(s)
    merged = merged2

    missing: list[str] = []
    short_clips: list[str] = []
    by_idx = {l.idx: l for l in kept}
    for s in merged:
        if s.label.startswith("b:"):
            slot = s.label[2:]
            got = broll.take(slot, s.dur, section=s.section)
            if got:
                s.kind, s.src, s.src_in = "broll", str(got[0]), got[1]
            else:
                s.kind, s.note = "slug", f"b-roll/{slot}"
                missing.append(f"b-roll: {slot} (s{s.section}, {s.dur:.1f}s)")
        else:
            cam = s.label
            ln = by_idx.get(s.line_idx)
            # Deliberately signed: the very first shot starts before its own
            # first word, and we want the real idle frames in front of the
            # line rather than the robot already mid-sentence at frame one.
            into_line = (s.start - ln.start) if ln else 0.0

            def offer(c: str):
                """(camera, path, in-point, how far it falls short) or None."""
                got = podcast.get((c, s.section))
                if got is None:
                    return None
                p, base = got
                if ln is not None and ln.src_t >= 0:
                    # Pull the frames that belong to this line. The offset
                    # inside the shot is preserved, so a J-cut or an internal
                    # cut lands on the right moment of the performance.
                    src_in = round(max(0.0, ln.src_t + into_line - base), 3)
                else:
                    src_in = 0.0
                d = probe(p)["dur"]
                return c, p, src_in, (round(src_in + s.dur - d, 3) if d else 0.0)

            # The wanted angle first, then anything else that covers the shot.
            # A per-section clip that stops at its section end cannot cover a
            # shot that runs into the gap, and holding a frame on a talking
            # robot reads as a freeze - so prefer an angle that has the frames.
            best = None
            for c in [cam] + [x for x in CAMERAS if x != cam]:
                r = offer(c)
                if r is None:
                    continue
                if r[3] <= 0.05:
                    best = r
                    break
                if best is None or r[3] < best[3]:
                    best = r

            if best is not None:
                used, p, src_in, short = best
                s.kind, s.src, s.src_in = "cam", str(p), src_in
                if used != cam:
                    s.note = f"wanted {cam}, used {used}"
                if short > 0.05:
                    s.note = (f"{s.note + '; ' if s.note else ''}"
                              f"{short:.2f}s short, last frame holds")
                    short_clips.append(f"{used}/s{s.section}")
            else:
                s.kind, s.note = "slug", f"cam {cam} / section {s.section}"
                missing.append(f"host cam {cam} section {s.section} ({s.dur:.1f}s)")
    if missing:
        log.append(f"placeholders: {len(missing)} shot(s) have no footage yet")
        for m in missing:
            log.append(f"  slug <- {m}")
    if short_clips:
        log.append(f"podcast: clip(s) end before the shot does: "
                   f"{sorted(set(short_clips))} (last frame will hold)")

    edl.shots = merged

    # --- dissolves --------------------------------------------------------
    refused = []
    for i in range(1, len(merged)):
        cur, prev = merged[i], merged[i - 1]
        if cur.section == prev.section or cur.section not in DISSOLVES:
            continue
        d = DISSOLVES[cur.section]
        if prev.dur <= d + 0.4 or cur.dur <= d + 0.4:
            refused.append(f"s{cur.section} (shot too short)")
            continue
        if prev.kind == "cam" and cur.kind == "cam":
            # same set, same robots, barely different framing: a dissolve here
            # double-exposes them and looks like a fault, not a transition
            refused.append(f"s{cur.section} (host-to-host would ghost)")
            continue
        edl.dissolves.append({"at": round(cur.start, 3), "dur": d, "section": cur.section})
    log.append(f"transitions: {len(edl.dissolves)} dissolve(s), "
               f"{len(merged) - 1 - len(edl.dissolves)} hard cut(s)"
               + (f"; dissolve refused at {refused}" if refused else ""))

    # --- lower thirds: on the cut that opens the section -------------------
    for sec, text in LOWER_THIRDS.items():
        cut = next((s for s in merged if s.section == sec), None)
        if cut is None:
            continue
        start = cut.start
        # if that shot dissolves in, hang the title until the dissolve is done
        for d in edl.dissolves:
            if abs(d["at"] - start) < 0.05:
                start += d["dur"]
        edl.titles.append(Title(start=round(start, 3), dur=LT_HOLD, text=text))
    log.append(f"lower-thirds: {len(edl.titles)}")

    # --- end card ---------------------------------------------------------
    endcard_start = round(narration_end + END_BEAT, 3)
    if merged:
        merged[-1].dur = round(max(0.4, endcard_start - merged[-1].start), 3)
    edl.shots.append(
        Shot(start=endcard_start, dur=ENDCARD_HOLD, kind="endcard", label="endcard",
             section=99, note="animacy / github / web")
    )
    edl.total = round(endcard_start + ENDCARD_HOLD + TAIL_BLACK, 3)

    # --- room tone under the host scenes only -----------------------------
    for s in edl.shots:
        if s.kind in ("cam", "slug") and s.label in CAMERAS:
            if edl.roomtone and abs(edl.roomtone[-1]["start"] + edl.roomtone[-1]["dur"] - s.start) < 0.05:
                edl.roomtone[-1]["dur"] = round(s.end - edl.roomtone[-1]["start"], 3)
            else:
                edl.roomtone.append({"start": s.start, "dur": s.dur})

    log.append(f"runtime: {mmss(edl.total)} total "
               f"({mmss(narration_end)} narration, {len(edl.shots)} shots, {len(kept)} lines)")
    return edl


def save_edl(edl: EDL) -> None:
    EDIT_DATA.mkdir(parents=True, exist_ok=True)
    EDL_JSON.write_text(json.dumps(edl.to_json(), indent=2), encoding="utf-8")
    BUILD_LOG.write_text("\n".join(edl.log) + "\n", encoding="utf-8")


def load_edl() -> EDL:
    blob = json.loads(EDL_JSON.read_text(encoding="utf-8"))
    edl = EDL(
        shots=[Shot(**s) for s in blob["shots"]],
        narration=[AudioCue(**a) for a in blob["narration"]],
        titles=[Title(**t) for t in blob["titles"]],
        dissolves=blob["dissolves"],
        roomtone=blob["roomtone"],
        lines=[Line(**l) for l in blob["lines"]],
        total=blob["total"],
        log=blob.get("log", []),
        dropped=blob.get("dropped", []),
    )
    return edl
