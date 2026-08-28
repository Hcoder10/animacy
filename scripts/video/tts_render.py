"""Render the two-host narration for the animacy demo film.

Reads ``docs/video/script.md``, finds every spoken line, and synthesises one WAV
per line with a distinct voice per host:

* **LAMP**   — British male, low and slow: dry, deadpan.
* **REACHY** — American male, higher and quicker: warm.

Output is 24 kHz mono PCM16 (plus exact 16 kHz copies under ``16k/`` for the
motion pipeline, which works at :data:`animacy.schema.AUDIO_SR`), silence
trimmed, and loudness-matched with ffmpeg ``loudnorm`` so no line jumps out in
the edit.

    python scripts/video/tts_render.py --script docs/video/script.md --out data/video/voice

Reproducible where it counts: on the default ``--device cpu`` a re-run gives
WAVs with identical sample counts, so every duration -- and any timeline built
from them -- comes back the same. Samples themselves can move by a few LSB
(measured worst case 4/32768, about -78 dBFS, inaudible); set OMP_NUM_THREADS=1
if you need byte-identical output too. ``--device cuda`` is ~2x faster but is
not duration-stable (a repeat pass moved total length by 1.5 s), so it is
opt-in.

``index`` in the manifest is 0-based, matching ``scripts/video/show_build.py``.
The ``NN_`` filename prefix is 1-based and is only there to sort the takes in
script order.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------

KOKORO_REPO = "hexgrad/Kokoro-82M"

#: Per-host voice. ``lang`` is Kokoro's pipeline code ('a' American, 'b' British).
#: ``speed`` is Kokoro's own duration scale, not a resample — pitch is untouched.
VOICES = {
    # ~96 Hz median f0, ~137 wpm. The lowest-pitched voice in the pack.
    "lamp": {"voice": "bm_lewis", "lang": "b", "speed": 0.90},
    # ~118 Hz median f0, ~166 wpm. Warm mid-range American.
    "reachy": {"voice": "am_michael", "lang": "a", "speed": 1.03},
}

TARGET_LUFS = -18.0
TARGET_TP_DBFS = -1.5
NATIVE_SR = 24000  # Kokoro's own output rate; no resampling on the master

# ---------------------------------------------------------------------------
# script parsing
# ---------------------------------------------------------------------------

RE_SECTION = re.compile(r"^##\s+(.+?)\s*$")
RE_LINE = re.compile(r"^\*\*(LAMP|REACHY):\*\*\s*(.+?)\s*$")


@dataclass
class Line:
    index: int
    host: str
    section: str
    text: str          # verbatim from the script
    text_spoken: str   # after TTS normalisation
    wav: str
    wav16k: str
    seconds: float
    voice: str
    speed: float
    peak_dbfs: float


def parse_script(path: str) -> list[tuple[str, str, str]]:
    """Return [(host, section, text)] in script order."""
    out: list[tuple[str, str, str]] = []
    section = ""
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            m = RE_SECTION.match(raw)
            if m:
                # "3 — One file per robot (over: ...)" -> "3 — One file per robot"
                section = m.group(1).split(" (over:")[0].strip()
                continue
            m = RE_LINE.match(raw)
            if m:
                out.append((m.group(1).lower(), section, m.group(2).strip()))
    return out


# ---------------------------------------------------------------------------
# text normalisation for TTS
# ---------------------------------------------------------------------------

#: Straight substitutions, checked against Kokoro's phonemiser output.
#: Each entry is (pattern, replacement, why).
SUBS = [
    # spelled out as R-E-A-D-M-E otherwise
    (r"\bREADME\b", "read me", "README is spelled letter by letter"),
    # ran together as one word: dIgri:QvfrI:d@m
    (r"\bdegree-of-freedom\b", "degree of freedom", "hyphens collapse the phrase"),
    # ri-TAR-git otherwise; the field says REE-target
    (r"\bRetarget\b", "Re-target", "wrong stress without the hyphen"),
    (r"\bretarget\b", "re-target", "wrong stress without the hyphen"),
]


def normalise(text: str) -> str:
    """Prepare a script line for the synthesiser.

    Em dashes carry the performance in this script, so they are translated
    rather than deleted: a trailing one leaves the line unfinished (no final
    punctuation, so the intonation stays open for the interruption), a leading
    one is dropped and the sentence re-capitalised, and an internal one becomes
    a comma.
    """
    t = text.strip()

    open_ended = False
    if t.endswith("—"):
        t = t[:-1].rstrip().rstrip(",")
        open_ended = True
    if t.startswith("—"):
        t = t[1:].lstrip()
        if t:
            t = t[0].upper() + t[1:]

    t = re.sub(r"\s*—\s*", ", ", t)          # internal em dash -> comma pause
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')

    for pat, rep, _why in SUBS:
        t = re.sub(pat, rep, t)

    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\s+", " ", t).strip()

    # an unfinished line must not end on a full stop
    if open_ended:
        t = t.rstrip(".")
    return t


def slug(text: str, max_len: int = 28) -> str:
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    stop = {"the", "a", "an", "and", "of", "is", "it", "that", "on", "in", "to", "so"}
    keep = [w for w in words if w not in stop] or words
    s = ""
    for w in keep:
        if s and len(s) + 1 + len(w) > max_len:
            break
        s = f"{s}-{w}" if s else w
    return s or "line"


# ---------------------------------------------------------------------------
# audio helpers
# ---------------------------------------------------------------------------

def trim_silence(x: np.ndarray, sr: int, pad_ms: float = 40.0,
                 fade_ms: float = 5.0) -> np.ndarray:
    """Trim head/tail silence, keep a short pad, and fade the edges."""
    if x.size == 0:
        return x
    win = max(1, int(sr * 0.010))
    n = len(x) // win
    if n == 0:
        return x
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    thresh = max(1e-4, 0.02 * float(rms.max()))
    voiced = np.flatnonzero(rms > thresh)
    if voiced.size == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    a = max(0, voiced[0] * win - pad)
    b = min(len(x), (voiced[-1] + 1) * win + pad)
    y = x[a:b].copy()
    f = min(int(sr * fade_ms / 1000.0), len(y) // 2)
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (needed for loudness matching)")
    return exe


def loudnorm(src: str, dst: str, sr: int) -> None:
    """Two-pass ffmpeg loudnorm to a fixed LUFS with a true-peak ceiling."""
    exe = _ffmpeg()
    filt = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP_DBFS}:LRA=11"
    probe = subprocess.run(
        [exe, "-hide_banner", "-nostats", "-i", src, "-af", f"{filt}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    measured = None
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", probe.stderr, re.S)
    if m:
        try:
            measured = json.loads(m[-1])
        except json.JSONDecodeError:
            measured = None
    if measured:
        filt += (f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                 f":measured_LRA={measured['input_lra']}"
                 f":measured_thresh={measured['input_thresh']}"
                 f":offset={measured['target_offset']}:linear=true")
    run = subprocess.run(
        [exe, "-hide_banner", "-nostats", "-y", "-i", src, "-af", filt,
         "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", dst],
        capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg loudnorm failed for {src}:\n{run.stderr[-800:]}")


def resample_copy(src: str, dst: str, sr: int) -> None:
    run = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-nostats", "-y", "-i", src,
         "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", dst],
        capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed for {src}:\n{run.stderr[-800:]}")


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------

class KokoroEngine:
    """Kokoro-82M (Apache-2.0). One KPipeline per language code, cached."""

    name = "kokoro"

    def __init__(self, seed: int = 0, device: str = "cpu"):
        import torch
        from kokoro import KPipeline  # noqa: F401  (import cost paid once)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._KPipeline = KPipeline
        self._pipes: dict[str, object] = {}
        self.device = device
        import kokoro
        self.version = getattr(kokoro, "__version__", "unknown")

    def _pipe(self, lang: str):
        if lang not in self._pipes:
            self._pipes[lang] = self._KPipeline(lang_code=lang, repo_id=KOKORO_REPO,
                                                device=self.device)
        return self._pipes[lang]

    def synth(self, text: str, cfg: dict) -> tuple[np.ndarray, int]:
        segs = list(self._pipe(cfg["lang"])(text, voice=cfg["voice"], speed=cfg["speed"]))
        if not segs:
            raise RuntimeError(f"kokoro produced no audio for: {text!r}")
        audio = np.concatenate([s.audio.numpy() for s in segs]).astype(np.float32)
        return audio, NATIVE_SR

    def meta(self) -> dict:
        return {"engine": "kokoro", "package_version": self.version,
                "repo_id": KOKORO_REPO, "params": "82M", "licence": "Apache-2.0",
                "offline": True, "device": self.device}


class SapiEngine:
    """Windows System.Speech fallback — no download, markedly more robotic."""

    name = "sapi"

    def __init__(self, seed: int = 0, device: str = "cpu"):
        self.device = device
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from animacy.tts import synth_sapi
        self._synth = synth_sapi

    def synth(self, text: str, cfg: dict) -> tuple[np.ndarray, int]:
        rate = int(round((cfg["speed"] - 1.0) * 10))
        return self._synth(text, rate=rate, voice=cfg.get("sapi_voice"))

    def meta(self) -> dict:
        return {"engine": "sapi", "package_version": "windows-system-speech",
                "repo_id": None, "licence": "OS-bundled", "offline": True,
                "device": "cpu"}


ENGINES = {"kokoro": KokoroEngine, "sapi": SapiEngine}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def man_path_of(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", default=os.path.join(root, "docs", "video", "script.md"))
    ap.add_argument("--out", default=os.path.join(root, "data", "video", "voice"))
    ap.add_argument("--engine", default="kokoro", choices=sorted(ENGINES))
    ap.add_argument("--voice-lamp", default=None, help="override the LAMP voice")
    ap.add_argument("--voice-reachy", default=None, help="override the REACHY voice")
    ap.add_argument("--speed-lamp", type=float, default=None)
    ap.add_argument("--speed-reachy", type=float, default=None)
    ap.add_argument("--sr", type=int, default=NATIVE_SR, help="master sample rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu is bit-reproducible; cuda is ~2x faster but its "
                         "LSTM kernels are not bit-identical between runs")
    ap.add_argument("--only", type=int, default=None,
                    help="render just this 0-based line index")
    ap.add_argument("--no-loudnorm", action="store_true")
    args = ap.parse_args()

    import soundfile as sf

    voices = {h: dict(v) for h, v in VOICES.items()}
    for host, vo, sp in (("lamp", args.voice_lamp, args.speed_lamp),
                         ("reachy", args.voice_reachy, args.speed_reachy)):
        if vo:
            voices[host]["voice"] = vo
            voices[host]["lang"] = "b" if vo[0] == "b" else "a"
        if sp:
            voices[host]["speed"] = sp

    lines = parse_script(args.script)
    if not lines:
        print(f"no spoken lines found in {args.script}", file=sys.stderr)
        return 2

    out_dir = os.path.abspath(args.out)
    raw_dir = os.path.join(out_dir, "_raw")
    d16_dir = os.path.join(out_dir, "16k")
    for d in (out_dir, raw_dir, d16_dir):
        os.makedirs(d, exist_ok=True)

    engine = ENGINES[args.engine](seed=args.seed, device=args.device)
    records: list[Line] = []
    t_start = time.time()

    # ``index`` is 0-based so it lines up with show_build.py's own line numbering;
    # the filename prefix stays 1-based so the takes sort in script order.
    for i, (host, section, text) in enumerate(lines):
        if args.only is not None and i != args.only:
            continue
        cfg = voices[host]
        spoken = normalise(text)
        name = f"{i + 1:02d}_{host}_{slug(text)}"
        raw_path = os.path.join(raw_dir, name + ".wav")
        final_path = os.path.join(out_dir, name + ".wav")
        d16_path = os.path.join(d16_dir, name + ".wav")

        audio, sr = engine.synth(spoken, cfg)
        audio = trim_silence(audio, sr)
        sf.write(raw_path, audio, sr, subtype="PCM_16")

        if args.no_loudnorm:
            resample_copy(raw_path, final_path, args.sr)
        else:
            loudnorm(raw_path, final_path, args.sr)
        resample_copy(final_path, d16_path, 16000)

        done, done_sr = sf.read(final_path, dtype="float32")
        peak = float(np.abs(done).max()) if done.size else 0.0
        peak_db = 20 * np.log10(peak) if peak > 0 else -120.0
        seconds = len(done) / done_sr

        records.append(Line(index=i, host=host, section=section, text=text,
                            text_spoken=spoken, wav=os.path.basename(final_path),
                            wav16k=f"16k/{os.path.basename(d16_path)}",
                            seconds=round(seconds, 3), voice=cfg["voice"],
                            speed=cfg["speed"], peak_dbfs=round(peak_db, 2)))
        print(f"[{i + 1:02d}/{len(lines)}] {host:6s} {seconds:5.2f}s "
              f"peak={peak_db:6.2f} dBFS  {name}", flush=True)

    # --only re-renders a single take; merge it into whatever the manifest
    # already holds so a one-line fix never truncates the other entries.
    man_path = man_path_of(out_dir)
    if args.only is not None and os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as fh:
            prev = json.load(fh).get("lines", [])
        merged = {int(e["index"]): e for e in prev}
        for r in records:
            merged[r.index] = asdict(r)
        kept = [merged[k] for k in sorted(merged)]
    else:
        kept = [asdict(r) for r in records]

    total = sum(float(e["seconds"]) for e in kept)
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": os.path.relpath(os.path.abspath(args.script), root).replace("\\", "/"),
        "engine": engine.meta(),
        "voices": {h: {"host": h, **voices[h]} for h in voices},
        "audio": {"sample_rate": args.sr, "channels": 1, "format": "pcm_s16le",
                  "sample_rate_16k_copies": 16000,
                  "loudness_target_lufs": TARGET_LUFS if not args.no_loudnorm else None,
                  "true_peak_ceiling_dbfs": TARGET_TP_DBFS if not args.no_loudnorm else None},
        "seed": args.seed,
        "device": args.device,
        "index_base": 0,
        "line_count": len(kept),
        "total_seconds": round(total, 3),
        "lines": kept,
    }
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\n{len(kept)} lines, {total:.1f} s "
          f"({total/60:.1f} min) in {time.time()-t_start:.0f} s wall")
    print(f"manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
