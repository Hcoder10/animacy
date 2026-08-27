"""Import clips recorded by the web viewer's Record mode into standard clip dirs.

    python -m animacy.import_browser <zip-or-dir> -o data/clips/<name>

A browser segment is three files:

* ``motion.json`` — ``HumanClip.to_web_json`` shape:
  ``{schema, rate_hz, n, channels, data: {channel: [...]}}``, ``null`` for NaN.
* ``audio.webm`` — MediaRecorder Opus, same clock as the frames (starts within
  a few ms of frame 0). ``.wav/.ogg/.mp4/.m4a`` are accepted too.
* ``meta.json`` — ``source: "webcam-browser"``, ``role: speaking|listening``,
  ``arm``, ``neutral``, ``license`` (CC-BY-4.0), ``versions``; optional
  ``audio_offset_s`` (positive = the audio started that long after frame 0).

Segments are found by locating every ``*motion.json`` under the input (a zip
is extracted first); ``<prefix>audio.<ext>`` and ``<prefix>meta.json`` next to
it belong to the same segment, so both ``seg_01/motion.json`` and
``seg_01_motion.json`` layouts work. One segment -> ``-o`` is the clip dir;
several -> ``-o/<segment>/``.

Per segment: parse (``null`` -> NaN, channels completed, ``t`` rebuilt on the
``rate_hz`` grid — the max deviation from the browser's own ``t`` is recorded);
decode the audio to 16 kHz mono (``ffmpeg`` on PATH, else imageio-ffmpeg's
binary; a ``.wav`` needs neither); apply ``audio_offset_s``; trim/pad the
audio to the motion duration and record the offset; ``speaking`` refilled by
the same VAD ``animacy capture`` uses (silero-vad, else energy+hysteresis)
for ``role: speaking``, forced to 0 for ``role: listening``; rows flagged
valid but carrying NaN are demoted to invalid (counted); the same light
zero-phase 8 Hz smoothing per contiguous valid run as capture; bounds clip;
``HumanClip.validate()``; save.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import capture_math as cm
from .schema import (ARM_CHANNELS, AUDIO_SR, BOUNDS, CHANNELS, FACE_CHANNELS, FLAGS, RATE_HZ,
                     TORSO_CHANNELS, HumanClip)

AUDIO_EXTS = (".webm", ".wav", ".ogg", ".opus", ".mp4", ".m4a", ".mp3", ".flac")
SMOOTH_CUTOFF_HZ = 8.0  # identical to animacy.capture.SMOOTH_CUTOFF_HZ (kept literal: capture.py is not imported here)


# ------------------------------------------------------------------ discovery
def find_segments(root: str) -> List[Dict[str, Optional[str]]]:
    """[{name, motion, audio, meta}, ...] for every ``*motion.json`` under ``root``."""
    segs = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith("motion.json"):
                continue
            prefix = f[: -len("motion.json")]
            audio, meta = None, None
            for g in files:
                gl = g.lower()
                if gl.startswith(prefix.lower()):
                    stem = gl[len(prefix):]
                    if stem.startswith("audio") and gl.endswith(AUDIO_EXTS) and audio is None:
                        audio = os.path.join(dirpath, g)
                    elif stem == "meta.json":
                        meta = os.path.join(dirpath, g)
            name = prefix.strip("_-. ") or os.path.basename(os.path.normpath(dirpath))
            segs.append({"name": name, "motion": os.path.join(dirpath, f), "audio": audio, "meta": meta})
    return segs


def _safe_name(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")
    return out or "segment"


# ------------------------------------------------------------------ parsing
def frames_from_web_json(obj: Dict) -> Tuple[pd.DataFrame, Dict]:
    """``to_web_json`` dict -> canonical frame table on the rate grid + parse notes."""
    rate = float(obj.get("rate_hz", RATE_HZ))
    data = obj.get("data", {}) or {}
    n = int(obj.get("n", 0) or 0)
    lengths = {c: len(v) for c, v in data.items() if v is not None}
    if not lengths:
        raise ValueError("motion.json has no data")
    if n <= 0:
        n = max(lengths.values())
    bad = {c: ln for c, ln in lengths.items() if ln != n}
    if bad:
        raise ValueError(f"motion.json channel lengths differ from n={n}: {bad}")
    notes: Dict = {"rate_hz": rate, "n": n, "unknown_channels": sorted(set(data) - set(CHANNELS)),
                   "missing_channels": [c for c in CHANNELS if c not in data]}
    df = pd.DataFrame(index=np.arange(n))
    for c in CHANNELS:
        if c in data:
            v = np.array([np.nan if x is None else float(x) for x in data[c]], dtype=np.float64)
        else:
            v = np.zeros(n) if (c in FLAGS or c == "t") else np.full(n, np.nan)
        df[c] = v
    t_browser = df["t"].to_numpy().copy()
    df["t"] = np.arange(n) / rate
    if "t" in data:
        dev = np.nanmax(np.abs(t_browser - df["t"].to_numpy())) if n else 0.0
        notes["t_max_dev_ms"] = float(dev * 1000.0) if np.isfinite(dev) else None
    else:
        notes["t_max_dev_ms"] = None
    for c in FLAGS:
        df[c] = (df[c].fillna(0.0).to_numpy() > 0.5).astype(np.float64)
    return df, notes


# ------------------------------------------------------------------ audio
def decode_audio(path: Optional[str], sr: int = AUDIO_SR) -> Tuple[Optional[np.ndarray], str]:
    """Any browser audio file -> float32 mono at ``sr``; (None, reason) if impossible."""
    if not path or not os.path.exists(path):
        return None, "no audio file"
    if path.lower().endswith(".wav"):
        import soundfile as sf

        data, got = sf.read(path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        if got != sr:
            from scipy.signal import resample_poly

            g = math.gcd(int(sr), int(got))
            mono = resample_poly(mono, sr // g, got // g).astype(np.float32)
        return mono, "soundfile (wav)"
    from .audio import extract_audio

    return extract_audio(path, sr)


def align_audio(audio: np.ndarray, n_frames: int, rate_hz: float, sr: int, offset_s: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """Shift by ``offset_s`` (audio started after frame 0 -> pad the front), then trim/pad to
    ``n_frames / rate_hz`` seconds (the audio spans the last frame's period, as capture does)."""
    want = int(round(n_frames / rate_hz * sr))
    shift = int(round(offset_s * sr))
    if shift > 0:
        audio = np.concatenate([np.zeros(shift, np.float32), audio])
    elif shift < 0:
        audio = audio[-shift:]
    have = len(audio)
    out = np.zeros(want, np.float32)
    k = min(have, want)
    out[:k] = audio[:k]
    return out, {"audio_offset_s": float(offset_s), "audio_len_s": have / sr, "motion_len_s": want / sr,
                 "audio_minus_motion_ms": float((have - want) / sr * 1000.0),
                 "action": "trimmed" if have > want else ("padded" if have < want else "exact")}


# ------------------------------------------------------------------ frames
def _demote_invalid(df: pd.DataFrame) -> Dict[str, int]:
    """face_valid/arm_valid rows whose channels carry NaN become invalid (validate() demands it)."""
    fixed = {}
    face_nan = df[FACE_CHANNELS].isna().any(axis=1).to_numpy()
    m = (df["face_valid"].to_numpy() > 0) & face_nan
    fixed["face_valid_demoted"] = int(m.sum())
    df.loc[m, "face_valid"] = 0.0
    arm_nan = df[ARM_CHANNELS].isna().any(axis=1).to_numpy()
    m = (df["arm_valid"].to_numpy() > 0) & arm_nan
    fixed["arm_valid_demoted"] = int(m.sum())
    df.loc[m, "arm_valid"] = 0.0
    return fixed


def smooth_like_capture(df: pd.DataFrame, rate_hz: float) -> pd.DataFrame:
    """capture.build_frames' smoothing: zero-phase 8 Hz per contiguous valid run, per group,
    NaN outside, then the schema's bounds clip."""
    df = df.copy()
    face_v = df["face_valid"].to_numpy() > 0
    arm_v = df["arm_valid"].to_numpy() > 0
    torso_v = ~df[TORSO_CHANNELS].isna().any(axis=1).to_numpy()
    for cols, valid in ((FACE_CHANNELS, face_v), (TORSO_CHANNELS, torso_v), (ARM_CHANNELS, arm_v)):
        vals = np.array(df[cols].to_numpy(dtype=np.float64), copy=True)  # to_numpy() may be read-only (CoW)
        vals[~valid] = np.nan
        sm = cm.smooth_runs(vals, valid, SMOOTH_CUTOFF_HZ, rate_hz)
        for j, c in enumerate(cols):
            lo, hi = BOUNDS[c]
            df[c] = np.clip(sm[:, j], lo, hi)
        df.loc[~valid, cols] = np.nan
    return df


def _stats(df: pd.DataFrame) -> Dict:
    st = {"n_frames": int(len(df)), "face_valid_frac": float(df["face_valid"].mean()) if len(df) else 0.0,
          "arm_valid_frac": float(df["arm_valid"].mean()) if len(df) else 0.0,
          "speaking_frac": float(df["speaking"].mean()) if len(df) else 0.0}
    for c in ("head_yaw", "head_pitch", "head_roll", "mouth_open"):
        v = df[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        st[f"{c}_std"] = float(v.std()) if len(v) else float("nan")
    return st


def _tool_versions() -> Dict[str, str]:
    from . import __version__

    out = {"animacy": __version__, "python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


# ------------------------------------------------------------------ one segment
def import_segment(motion_path: str, audio_path: Optional[str], meta_path: Optional[str], out_dir: str,
                   vad_fn: Optional[Callable[[Optional[np.ndarray], int, np.ndarray], Tuple[np.ndarray, str]]] = None,
                   smooth: bool = True) -> HumanClip:
    """Build, validate and save one clip dir. ``vad_fn`` defaults to ``animacy.vad.speaking_mask``."""
    obj = json.load(open(motion_path, encoding="utf-8"))
    df, notes = frames_from_web_json(obj)
    rate = notes["rate_hz"]
    bmeta = json.load(open(meta_path, encoding="utf-8")) if meta_path and os.path.exists(meta_path) else {}
    role = str(bmeta.get("role", "")).lower()
    browser_speaking_frac = float(df["speaking"].mean()) if len(df) else 0.0

    audio, audio_backend = decode_audio(audio_path)
    align: Dict = {}
    if audio is not None:
        audio, align = align_audio(audio, len(df), rate, AUDIO_SR, float(bmeta.get("audio_offset_s", 0.0) or 0.0))

    if vad_fn is None:
        from .vad import speaking_mask as vad_fn  # type: ignore[assignment]
    if role == "listening":
        df["speaking"] = 0.0
        vad_backend = "forced 0 (role=listening)"
    else:
        mask, vad_backend = vad_fn(audio, AUDIO_SR, df["t"].to_numpy(dtype=float))
        df["speaking"] = np.asarray(mask, dtype=float)
        if role != "speaking":
            vad_backend += " (role unknown -> VAD)"

    demoted = _demote_invalid(df)
    if smooth:
        df = smooth_like_capture(df, rate)
    df = df[CHANNELS].astype(np.float32)

    meta = {
        "source": bmeta.get("source", "webcam-browser"),
        "source_path": os.path.abspath(motion_path),
        "role": role or None,
        "arm": bmeta.get("arm", "right"),
        "neutral": bmeta.get("neutral"),
        "license": bmeta.get("license", "CC-BY-4.0"),
        "subject": bmeta.get("subject", "self"),
        "rate_hz": rate,
        "browser": {k: v for k, v in bmeta.items() if k not in ("neutral",)},
        "tool_versions": {"importer": _tool_versions(), "browser": bmeta.get("versions")},
        "vad": vad_backend,
        "audio_backend": audio_backend,
        "audio_align": align,
        "parse": notes,
        "fixes": demoted,
        "smoothing": {"kind": "zero-phase butterworth order 2 per contiguous valid run", "cutoff_hz": SMOOTH_CUTOFF_HZ} if smooth else None,
        "browser_speaking_frac": browser_speaking_frac,
        "stats": _stats(df),
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    clip = HumanClip.from_frames(df, **meta)
    clip.audio = audio
    clip.sr = AUDIO_SR
    clip.save(out_dir)
    return clip


def report(name: str, clip: HumanClip, out_dir: str) -> List[str]:
    probs = clip.validate()
    st, al = clip.meta["stats"], clip.meta.get("audio_align") or {}
    print(f"{name}: {len(clip)} frames ({clip.duration:.1f}s) -> {out_dir}  validate={'OK' if not probs else probs}")
    print(f"   role={clip.meta['role']}  face_valid {st['face_valid_frac']:.0%}  arm_valid {st['arm_valid_frac']:.0%}  "
          f"speaking {st['speaking_frac']:.0%} [{clip.meta['vad']}]  head_yaw std {st['head_yaw_std']:.1f} deg")
    if al:
        print(f"   audio {al['audio_len_s']:.2f}s vs motion {al['motion_len_s']:.2f}s: {al['audio_minus_motion_ms']:+.0f} ms -> {al['action']}"
              f" (offset {al['audio_offset_s'] * 1000:+.0f} ms) [{clip.meta['audio_backend']}]")
    else:
        print(f"   no audio ({clip.meta['audio_backend']})")
    fx = clip.meta["fixes"]
    if any(fx.values()):
        print(f"   demoted invalid rows: {fx}")
    return probs


# ------------------------------------------------------------------ entry
def import_path(src: str, out: str, vad_fn=None, smooth: bool = True) -> Tuple[int, List[str]]:
    """Import a zip or directory. Returns (rc, clip dirs)."""
    tmp = None
    root = src
    if os.path.isfile(src) and zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp(prefix="animacy_import_")
        with zipfile.ZipFile(src) as z:
            for m in z.namelist():  # refuse path traversal
                if os.path.isabs(m) or ".." in m.replace("\\", "/").split("/"):
                    raise ValueError(f"unsafe zip member {m!r}")
            z.extractall(tmp)
        root = tmp
    elif not os.path.isdir(src):
        raise SystemExit(f"{src} is neither a directory nor a zip")
    try:
        segs = find_segments(root)
        if not segs:
            print(f"no *motion.json found under {src}")
            return 1, []
        rc, dirs = 0, []
        multi = len(segs) > 1
        for seg in segs:
            out_dir = os.path.join(out, _safe_name(seg["name"])) if multi else out
            try:
                clip = import_segment(seg["motion"], seg["audio"], seg["meta"], out_dir, vad_fn=vad_fn, smooth=smooth)
            except Exception as exc:  # noqa: BLE001
                print(f"{seg['name']}: FAILED {type(exc).__name__}: {exc}")
                rc = 1
                continue
            if report(seg["name"], clip, out_dir):
                rc = 1
            dirs.append(out_dir)
        return rc, dirs
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animacy.import_browser", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="zip file or directory from the web viewer's Record mode")
    p.add_argument("-o", "--output", required=True, help="clip dir (one segment) or parent dir (several)")
    p.add_argument("--no-smooth", action="store_true", help="keep the browser's raw values")
    return p


def run_from_args(a) -> int:
    rc, _ = import_path(a.src, a.output, smooth=not a.no_smooth)
    return rc


def main(argv=None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
