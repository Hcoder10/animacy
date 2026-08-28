"""docs/video/script.md + the voice takes -> one 30 Hz show timeline for the podcast set.

    python scripts/video/show_build.py                      # real voice manifest, or SAPI placeholder
    python scripts/video/show_build.py --placeholder         # force the placeholder voice

Every line of the script is spoken by one host. The SPEAKING host's motion comes
from ``animacy.serve.retrieval_motion`` driven by that line's own audio; the
LISTENING host's comes from the same audio in listen mode (``listen=True`` ->
speaking = 0, causal) with a constant gaze offset toward the speaker, applied
exactly as the viewer's listen-mode overlay applies it
(``head_yaw += GAZE_WEIGHT * g_yaw``, see web/js/talk.js). Both then go through
that robot's own ``ROBOT.md`` via ``animacy.retarget.retarget_clip``.

Nothing is keyframed. The only hand-authored numbers here are the *set*: how
long the settle between lines is, and how far each host is turned toward the
other. Between sections both hosts blend back to their profile rest pose.

Output (``data/video/podcast/``):
  show.json      the whole timeline: per-line joint tables + a global per-frame
                 track for both robots, so frame i of the video is row i here
  narration.wav  the takes concatenated with exactly the same gaps, 16 kHz mono

Because both come off the same integer frame clock, audio and motion cannot
drift: line k starts at sample ``f_start / fps * sr`` and at row ``f_start``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from animacy.profile import Profile, find_robot          # noqa: E402
from animacy.retarget import retarget_clip                # noqa: E402
from animacy.schema import BOUNDS, HumanClip              # noqa: E402
from animacy.serve import retrieval_motion                # noqa: E402

FPS = 30
SR = 16000

# --- the set (the only hand-authored numbers in this file) -------------------
# Gaps are integer frames so the audio clock and the motion clock are the same
# clock: 350 ms of settle is 11 frames at 30 fps (366.7 ms).
LINE_GAP_FRAMES = 11        # settle between two lines of the same section
SECTION_GAP_FRAMES = 33     # beat between sections: ease to rest, hold, ease out
LEAD_IN_FRAMES = 36         # both hosts at rest before the first line
TAIL_FRAMES = 45            # ... and after the last one
# Listen-mode gaze: the raw target handed to the overlay, in canonical degrees.
# The overlay applies GAZE_WEIGHT of it (web/js/talk.js), so the effective head
# turn is half of `gaze_yaw`.
#
# `gaze_yaw` differs per host because the same canonical degree buys a different
# amount of *looking* on each body: the lamp's head_yaw drives base_yaw at gain
# -1.37 AND wrist_roll at -1.99, and rolling a shade whose axis is off-centre
# swings its gaze too, so a lamp turns about three times as far per canonical
# degree as a reachy does. These land both hosts around 25-30 deg off the other
# when listening: clearly attending to them, still open to camera.
#
# `gaze_sign` is which way is "toward the other host" in canonical yaw. It is a
# measured fact, not a derived one: the two hosts need OPPOSITE canonical signs
# even though they face each other, because the lamp's head_yaw gain is negative
# and the reachy's is positive. The check is podcast.js `measure()` -> offAxisDeg,
# which must go DOWN while a host is listening; it caught both signs the wrong
# way round on the first pass. Re-run it if either ROBOT.md changes.
GAZE_WEIGHT = 0.5           # == talk.js GAZE_WEIGHT
LISTEN_GAZE_PITCH = 4.0     # -> +2 deg: the listener looks very slightly up at the speaker

HOSTS = {
    "LAMP": {"robot": "lamp", "key": "lamp", "gaze_sign": +1.0, "gaze_yaw": 16.0},
    "REACHY": {"robot": "reachy_mini", "key": "reachy", "gaze_sign": -1.0, "gaze_yaw": 28.0},
}
KEYS = ["lamp", "reachy"]


# ---------------------------------------------------------------- the script
def parse_script(path: str) -> Tuple[List[Dict], List[Dict]]:
    """``docs/video/script.md`` -> (sections, lines). A section is ``## N - Title``;
    a line is ``**HOST:** text``. Returns lines in script order with their section."""
    sections: List[Dict] = []
    lines: List[Dict] = []
    sec_re = re.compile(r"^##\s+(\S+)\s*[-—]\s*(.+?)\s*$")
    line_re = re.compile(r"^\*\*(" + "|".join(HOSTS) + r"):\*\*\s*(.+?)\s*$")
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            m = sec_re.match(raw.rstrip("\n"))
            if m:
                title = m.group(2)
                over = ""
                mo = re.search(r"\(over:\s*(.+?)\)\s*$", title)
                if mo:
                    over = mo.group(1)
                    title = title[: mo.start()].strip()
                sections.append({"index": len(sections), "number": m.group(1), "title": title,
                                 "over": over, "line_indices": []})
                continue
            m = line_re.match(raw.rstrip("\n"))
            if m and sections:
                host = m.group(1)
                text = re.sub(r"\s+", " ", m.group(2)).strip()
                sections[-1]["line_indices"].append(len(lines))
                lines.append({"index": len(lines), "section": sections[-1]["index"],
                              "host": host, "robot": HOSTS[host]["robot"], "text": text})
    if not lines:
        raise SystemExit(f"no spoken lines found in {path}")
    return sections, lines


# ---------------------------------------------------------------- the voice
def _wav_key(d: Dict) -> Optional[str]:
    for k in ("wav", "file", "path", "audio", "filename"):
        if isinstance(d.get(k), str) and d[k]:
            return d[k]
    return None


def load_voice_manifest(path: str, lines: List[Dict]) -> List[str]:
    """The ``voice`` agent's manifest -> one absolute wav path per script line.

    Tolerant on purpose: the manifest may be ``{"lines": [...]}`` or a bare list,
    and the wav may sit under any of a few obvious keys. Entries are matched by
    an explicit ``index`` when present, else by position in script order."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    items = obj if isinstance(obj, list) else (obj.get("lines") or obj.get("takes") or obj.get("items") or [])
    if not items:
        raise SystemExit(f"{path}: no line entries found")
    base = os.path.dirname(os.path.abspath(path))
    by_index: Dict[int, Dict] = {}
    for pos, it in enumerate(items):
        idx = it.get("index", it.get("line", pos))
        by_index[int(idx)] = it
    out: List[str] = []
    for ln in lines:
        it = by_index.get(ln["index"])
        if it is None:
            raise SystemExit(f"{path}: no audio for line {ln['index']} ({ln['host']}: {ln['text'][:40]}...)")
        w = _wav_key(it)
        if w is None:
            raise SystemExit(f"{path}: line {ln['index']} has no wav path (keys: {sorted(it)})")
        p = w if os.path.isabs(w) else os.path.join(base, w)
        if not os.path.exists(p):
            raise SystemExit(f"{path}: line {ln['index']} wav missing: {p}")
        said = re.sub(r"[^a-z0-9]+", " ", str(it.get("text", "")).lower()).strip()
        want = re.sub(r"[^a-z0-9]+", " ", ln["text"].lower()).strip()
        if said and want and said[:40] != want[:40]:
            print(f"[warn] line {ln['index']}: manifest text starts {said[:40]!r}, script has {want[:40]!r}")
        out.append(p)
    return out


def build_placeholder_voice(lines: List[Dict], out_dir: str) -> List[str]:
    """Stand-in takes from the local TTS so the set, the framing and the motion
    are all real while the ``voice`` agent's takes are still rendering. Rebuild
    the show against the real manifest when it lands."""
    import soundfile as sf

    from animacy.tts import synth

    os.makedirs(out_dir, exist_ok=True)
    voices = _sapi_voices()
    per_host = {"LAMP": voices[0] if voices else None,
                "REACHY": (voices[1] if len(voices) > 1 else (voices[0] if voices else None))}
    paths = []
    for ln in lines:
        p = os.path.join(out_dir, f"line_{ln['index']:03d}.wav")
        if not os.path.exists(p):
            wav, sr = _synth_voiced(ln["text"], per_host[ln["host"]], synth)
            sf.write(p, wav.astype(np.float32), sr)
        paths.append(p)
        print(f"  [voice] line {ln['index']:2d} {ln['host']:<7} {os.path.basename(p)}", flush=True)
    return paths


def _sapi_voices() -> List[str]:
    """Installed SAPI voice names, so the two hosts do not sound identical."""
    if sys.platform != "win32":
        return []
    import subprocess

    cmd = ("Add-Type -AssemblyName System.Speech; "
           "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
           "ForEach-Object { $_.VoiceInfo.Name }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001
        return []


def _synth_voiced(text: str, voice: Optional[str], synth):
    from animacy.tts import synth_sapi

    if voice and sys.platform == "win32":
        return synth_sapi(text, voice=voice)
    return synth(text)


def read_wav_16k(path: str) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


# ---------------------------------------------------------------- the motion
def apply_gaze(clip, yaw_deg: float, pitch_deg: float):
    """The viewer's listen-mode gaze overlay, offline (web/js/talk.js `_applyGaze`):
    a constant look direction blended *under* the generated motion, then clamped
    to the canonical sanity bounds. Returns the clip (modified in place)."""
    f = clip.frames
    for ch, add in (("head_yaw", GAZE_WEIGHT * yaw_deg), ("head_pitch", GAZE_WEIGHT * pitch_deg)):
        lo, hi = BOUNDS[ch]
        v = np.nan_to_num(f[ch].to_numpy(dtype=np.float64), nan=0.0) + add
        f[ch] = np.clip(v, lo, hi).astype(np.float32)
    return clip


def line_motion(wav: np.ndarray, text: str, checkpoint: str, speaking_host: str, seed: int) -> Dict[str, HumanClip]:
    """One line of audio -> a canonical clip for each host: the speaker's from
    talk mode, the listener's from listen mode plus the gaze offset."""
    out = {}
    for host, spec in HOSTS.items():
        listen = host != speaking_host
        clip = retrieval_motion(wav, SR, checkpoint=checkpoint, seed=seed, listen=listen, intent=text)
        if listen:
            apply_gaze(clip, spec["gaze_sign"] * spec["gaze_yaw"], LISTEN_GAZE_PITCH)
        probs = clip.validate()
        if probs:
            raise RuntimeError(f"invalid clip for {host} on {text[:40]!r}: {probs}")
        out[spec["key"]] = clip
    return out


def table_to_frames(table: pd.DataFrame, joints: List[str], n: int) -> np.ndarray:
    """A ``t``-indexed joint table -> exactly ``n`` rows on the 30 Hz grid.

    ``retarget_clip`` stretches time wherever a joint would break its speed
    ceiling, so its table can be a frame or two longer than the audio. ``n`` is
    the max over both robots and the audio, and ``np.interp`` holds the end
    value, so no motion is ever cut and the pose simply rests into the gap."""
    t = table["t"].to_numpy(dtype=np.float64)
    t = t - t[0]
    tn = np.arange(n) / FPS
    return np.stack([np.interp(tn, t, table[c].to_numpy(dtype=np.float64)) for c in joints], axis=1)


# ---------------------------------------------------------------- the timeline
def smoothstep(n: int) -> np.ndarray:
    """``n`` interior points of a 0->1 smoothstep (never 0, never 1: the endpoints
    are the frames on either side of the gap)."""
    u = (np.arange(n) + 1.0) / (n + 1.0)
    return u * u * (3.0 - 2.0 * u)


def blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """``n`` frames easing from pose ``a`` to pose ``b``."""
    s = smoothstep(n)[:, None]
    return a[None, :] * (1.0 - s) + b[None, :] * s


def section_gap(a: np.ndarray, rest: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Ease to the attentive rest pose, hold it, ease into the next line."""
    n_in = max(1, round(n * 0.36))
    n_out = max(1, round(n * 0.36))
    n_hold = max(0, n - n_in - n_out)
    return np.concatenate([blend(a, rest, n_in), np.repeat(rest[None, :], n_hold, axis=0), blend(rest, b, n_out)])


def build(args) -> int:
    script = os.path.join(ROOT, "docs", "video", "script.md")
    sections, lines = parse_script(script)
    print(f"[script] {len(sections)} sections, {len(lines)} lines")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    voice_manifest = os.path.abspath(args.voice)
    placeholder = args.placeholder or not os.path.exists(voice_manifest)
    if placeholder:
        if not args.placeholder:
            print(f"[voice] {voice_manifest} not there yet -> local TTS placeholder takes")
        wavs = build_placeholder_voice(lines, os.path.join(out_dir, "placeholder_voice"))
    else:
        wavs = load_voice_manifest(voice_manifest, lines)
        print(f"[voice] {len(wavs)} takes from {voice_manifest}")

    profiles: Dict[str, Profile] = {}
    joints: Dict[str, List[str]] = {}
    rest: Dict[str, np.ndarray] = {}
    for spec in HOSTS.values():
        p = find_robot(spec["robot"])
        profiles[spec["key"]] = p
        joints[spec["key"]] = [j.name for j in p.joints]
        rest[spec["key"]] = np.array([j.rest for j in p.joints], dtype=np.float64)
        if abs(p.rate_hz - FPS) > 1e-6:
            raise SystemExit(f"{spec['robot']}: rate_hz is {p.rate_hz}, the show clock is {FPS}")

    # --- per line: audio -> motion -> joints, on one frame count -------------
    per_line: List[Dict] = []
    for ln, wp in zip(lines, wavs):
        wav = read_wav_16k(wp)
        clips = line_motion(wav, ln["text"], args.checkpoint, ln["host"], seed=args.seed + ln["index"])
        tables = {k: retarget_clip(clips[k], profiles[k]) for k in KEYS}
        n = max([int(math.ceil(len(wav) / SR * FPS))] + [len(tables[k]) for k in KEYS])
        per_line.append({
            "line": ln, "wav_path": wp, "audio": wav, "n": n,
            "frames": {k: table_to_frames(tables[k], joints[k], n) for k in KEYS},
        })
        print(f"  [motion] line {ln['index']:2d} {ln['host']:<7} {len(wav)/SR:5.2f}s -> {n:4d} frames", flush=True)

    # --- assemble one clock -------------------------------------------------
    track: Dict[str, List[np.ndarray]] = {k: [] for k in KEYS}
    cursor = 0

    def emit(block: Dict[str, np.ndarray]) -> int:
        nonlocal cursor
        n = len(next(iter(block.values())))
        for k in KEYS:
            track[k].append(block[k])
        cursor += n
        return n

    emit({k: np.repeat(rest[k][None, :], LEAD_IN_FRAMES, axis=0) for k in KEYS})
    for i, item in enumerate(per_line):
        if i:
            prev = per_line[i - 1]
            same_section = item["line"]["section"] == prev["line"]["section"]
            g = LINE_GAP_FRAMES if same_section else SECTION_GAP_FRAMES
            gap = {}
            for k in KEYS:
                a, b = prev["frames"][k][-1], item["frames"][k][0]
                gap[k] = blend(a, b, g) if same_section else section_gap(a, rest[k], b, g)
            emit(gap)
        item["f_start"] = cursor
        emit(item["frames"])
    last = per_line[-1]["frames"]
    n_out = max(1, round(TAIL_FRAMES * 0.5))
    emit({k: np.concatenate([blend(last[k][-1], rest[k], n_out),
                             np.repeat(rest[k][None, :], TAIL_FRAMES - n_out, axis=0)]) for k in KEYS})

    tracks = {k: np.concatenate(track[k], axis=0) for k in KEYS}
    n_frames = cursor
    for k in KEYS:
        assert len(tracks[k]) == n_frames, (k, len(tracks[k]), n_frames)
    print(f"[show] {n_frames} frames = {n_frames / FPS:.2f}s at {FPS} fps")

    # --- narration on the same clock ---------------------------------------
    import soundfile as sf

    narration = np.zeros(int(round(n_frames / FPS * SR)) + SR // 10, dtype=np.float32)
    for item in per_line:
        s = int(round(item["f_start"] / FPS * SR))
        a = item["audio"]
        narration[s:s + len(a)] += a[: max(0, len(narration) - s)]
    peak = float(np.max(np.abs(narration))) if len(narration) else 0.0
    if peak > 0.99:
        narration *= 0.99 / peak
    wav_path = os.path.join(out_dir, "narration.wav")
    sf.write(wav_path, narration, SR)
    print(f"[show] {wav_path}  {len(narration) / SR:.2f}s peak {peak:.2f}")

    # --- show.json ----------------------------------------------------------
    def rows(a: np.ndarray) -> List[List[float]]:
        return np.round(a, 3).tolist()

    show = {
        "schema": "animacy.podcast.v1",
        "fps": FPS,
        "sr": SR,
        "n_frames": n_frames,
        "seconds": round(n_frames / FPS, 3),
        "narration_wav": "narration.wav",
        "placeholder_voice": bool(placeholder),
        "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), ROOT).replace("\\", "/"),
        "source": "retrieval",
        "set": {
            "line_gap_frames": LINE_GAP_FRAMES, "section_gap_frames": SECTION_GAP_FRAMES,
            "lead_in_frames": LEAD_IN_FRAMES, "tail_frames": TAIL_FRAMES,
            "gaze_weight": GAZE_WEIGHT, "listen_gaze_pitch": LISTEN_GAZE_PITCH,
            "listen_gaze_yaw": {s["key"]: s["gaze_yaw"] for s in HOSTS.values()},
        },
        "hosts": {spec["key"]: {"script_name": host, "robot": spec["robot"],
                                "display_name": profiles[spec["key"]].display_name,
                                "joints": joints[spec["key"]],
                                "rest": [round(v, 4) for v in rest[spec["key"]].tolist()]}
                  for host, spec in HOSTS.items()},
        "tracks": {k: {"joints": joints[k], "values": rows(tracks[k])} for k in KEYS},
        "sections": [],
        "lines": [],
    }
    for item in per_line:
        ln = item["line"]
        e = {"index": ln["index"], "section": ln["section"], "host": ln["host"],
             "robot": ln["robot"], "text": ln["text"],
             "wav": os.path.relpath(item["wav_path"], out_dir).replace("\\", "/"),
             "f_start": item["f_start"], "f_count": item["n"],
             "t_start": round(item["f_start"] / FPS, 3),
             "seconds": round(item["n"] / FPS, 3),
             "audio_seconds": round(len(item["audio"]) / SR, 3)}
        for k in KEYS:
            e[k] = {"joints": joints[k], "values": rows(item["frames"][k])}
        show["lines"].append(e)
    for sec in sections:
        idx = sec["line_indices"]
        if not idx:
            continue
        first, last_i = per_line[idx[0]], per_line[idx[-1]]
        f0 = first["f_start"]
        f1 = last_i["f_start"] + last_i["n"]
        show["sections"].append({"index": sec["index"], "number": sec["number"], "title": sec["title"],
                                 "over": sec["over"], "line_indices": idx,
                                 "f_start": f0, "f_end": f1,
                                 "t_start": round(f0 / FPS, 3), "t_end": round(f1 / FPS, 3)})

    out_json = os.path.join(out_dir, "show.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(show, fh, separators=(",", ":"))
    print(f"[show] {out_json}  {os.path.getsize(out_json) / 1e6:.2f} MB")
    if placeholder:
        print("[show] NOTE: placeholder voice. Re-run once data/video/voice/manifest.json exists.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--voice", default=os.path.join(ROOT, "data", "video", "voice", "manifest.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "video", "podcast"))
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "v2a"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--placeholder", action="store_true", help="force the local-TTS placeholder takes")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
