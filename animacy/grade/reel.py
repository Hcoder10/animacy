"""Blind numbering, seeded shuffle, reel assembly and the sealed manifest.

Per robot, every clip (candidates and calibration clips alike) gets a blind
number 1..N in a seeded random order; reels are consecutive runs of those
numbers so the judge can be told "this file holds clips 13 to 24". The map
number -> (source, seed, movement, vendor clip) is the SEALED manifest,
written next to the run outputs and never into the judge's workspace.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..profile import Profile, find_robot
from .movements import ClipSpec
from .render import FPS, ViewerRenderer, concat_mp4, encode_frames

CARD_SECONDS = 1.0
GAP_SECONDS = 0.5
MAX_REEL_SECONDS = 90.0
STRIP_HEIGHT = 28


@dataclass
class Numbered:
    number: int
    clip: ClipSpec


@dataclass
class Reel:
    robot: str
    index: int                               # 1-based per robot
    entries: List[Numbered]
    path: Optional[str] = None
    seconds: float = 0.0
    render: Dict = field(default_factory=dict)

    @property
    def numbers(self) -> List[int]:
        return [e.number for e in self.entries]

    @property
    def name(self) -> str:
        return f"{self.robot}_reel{self.index}"


def number_and_shuffle(clips: Sequence[ClipSpec], seed: int) -> List[Numbered]:
    """Seeded random order, then numbers 1..N in that order (same seed -> same order)."""
    order = list(clips)
    random.Random(seed).shuffle(order)
    return [Numbered(i + 1, c) for i, c in enumerate(order)]


def clip_seconds(clip: ClipSpec, card_seconds: float = CARD_SECONDS, gap_seconds: float = GAP_SECONDS) -> float:
    return card_seconds + clip.duration + gap_seconds


def chunk_reels(numbered: Sequence[Numbered], robot: str, max_seconds: float = MAX_REEL_SECONDS,
                card_seconds: float = CARD_SECONDS, gap_seconds: float = GAP_SECONDS) -> List[Reel]:
    """Split into as few reels as keep each under ``max_seconds``, balanced by duration."""
    secs = [clip_seconds(e.clip, card_seconds, gap_seconds) for e in numbered]
    total = float(sum(secs))
    n_reels = max(1, int(np.ceil(total / max_seconds)))
    target = total / n_reels
    reels: List[Reel] = []
    cur: List[Numbered] = []
    acc = 0.0
    for e, s in zip(numbered, secs):
        if cur and acc + s > target * 1.02 and len(reels) < n_reels - 1:
            reels.append(Reel(robot, len(reels) + 1, cur, seconds=acc))
            cur, acc = [], 0.0
        cur.append(e)
        acc += s
    if cur:
        reels.append(Reel(robot, len(reels) + 1, cur, seconds=acc))
    return reels


def plan_reels(clips: Sequence[ClipSpec], seed: int, max_seconds: float = MAX_REEL_SECONDS) -> Dict[str, List[Reel]]:
    """Per robot: numbered, shuffled, chunked (nothing rendered yet)."""
    out: Dict[str, List[Reel]] = {}
    for robot in sorted({c.robot for c in clips}):
        mine = [c for c in clips if c.robot == robot]
        numbered = number_and_shuffle(mine, seed + sum(map(ord, robot)))
        out[robot] = chunk_reels(numbered, robot, max_seconds)
    return out


def sealed_manifest(plans: Dict[str, List[Reel]], seed: int, extra: Optional[Dict] = None) -> Dict:
    m: Dict = {"schema": "animacy.grading.manifest.v1", "seed": seed, "robots": {}}
    if extra:
        m.update(extra)
    for robot, reels in plans.items():
        entry = {"clips": {}, "reels": []}
        for r in reels:
            entry["reels"].append({"index": r.index, "name": r.name, "numbers": r.numbers, "seconds": round(r.seconds, 2),
                                   "path": r.path})
            for e in r.entries:
                entry["clips"][str(e.number)] = e.clip.public()
        m["robots"][robot] = entry
    return m


def write_sealed_manifest(plans: Dict[str, List[Reel]], run_dir: str, seed: int, extra: Optional[Dict] = None) -> str:
    path = os.path.join(run_dir, "manifest_sealed.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sealed_manifest(plans, seed, extra), fh, indent=1)
    return path


# ---------------------------------------------------------------- speech strip
def speech_envelope(audio: Optional[np.ndarray], sr: int, n_frames: int, fps: float = FPS) -> np.ndarray:
    """Per-video-frame RMS of the utterance, normalised to [0, 1]; zeros when silent."""
    env = np.zeros(n_frames, dtype=np.float64)
    if audio is None or not len(audio):
        return env
    a = np.asarray(audio, dtype=np.float64)
    hop = int(round(sr / fps))
    for i in range(n_frames):
        seg = a[i * hop:(i + 1) * hop]
        if len(seg):
            env[i] = np.sqrt(np.mean(seg ** 2))
    if env.max() > 0:
        env = env / env.max()
    return env


def overlay_speech_strip(frames: List[bytes], env: np.ndarray, height: int = STRIP_HEIGHT) -> List[bytes]:
    """Burn a loudness strip (whole utterance, static) with a moving marker into every frame.

    The judge cannot hear the audio track (measured in the probe), so this is how it sees when the
    robot is speaking. Silent clips get a flat strip: the strip itself carries no origin information."""
    from io import BytesIO

    from PIL import Image, ImageDraw

    out: List[bytes] = []
    n = len(frames)
    for i, png in enumerate(frames):
        im = Image.open(BytesIO(png)).convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        y0 = h - height
        dr.rectangle([0, y0, w, h], fill=(14, 17, 23))
        dr.line([(0, h - 2), (w, h - 2)], fill=(60, 66, 80), width=1)
        if n > 1:
            for k in range(n):
                x = int(k * (w - 1) / (n - 1))
                bar = int(env[k] * (height - 6))
                col = (255, 184, 107) if k <= i else (110, 90, 60)
                if bar > 0:
                    dr.line([(x, h - 3), (x, h - 3 - bar)], fill=col, width=1)
            x = int(i * (w - 1) / (n - 1))
            dr.line([(x, y0), (x, h)], fill=(230, 235, 245), width=2)
        buf = BytesIO()
        im.save(buf, format="PNG", compress_level=3)
        out.append(buf.getvalue())
    return out


# ---------------------------------------------------------------- rendering
def slow_table(table, speed: float):
    """Stretch a joint table's ``t`` by 1/speed (0.5 = half speed): the renderer then samples twice as many frames."""
    out = table.copy()
    out["t"] = out["t"] / speed
    return out


def slow_audio(audio: Optional[np.ndarray], speed: float) -> Optional[np.ndarray]:
    """Time-stretch by plain resampling (pitch drops; the judge cannot hear it, the loudness strip is what matters)."""
    if audio is None or speed == 1.0:
        return audio
    from scipy.signal import resample_poly

    num, den = 1000, int(round(1000 * speed))
    return resample_poly(np.asarray(audio, np.float64), num, den).astype(np.float32)


def render_entry(renderer: ViewerRenderer, e: Numbered, profile: Profile, out_mp4: str, card_seconds: float = CARD_SECONDS,
                 speech_strip: bool = True, work_dir: Optional[str] = None, speed: float = 1.0) -> Dict:
    """One numbered clip -> MP4 (card + frames [+ strip] + audio). ``speed`` < 1 is the slow-motion variant:
    the card says "slow motion" (and nothing about why) so the judge does not score the slowness as intent."""
    table = e.clip.table if speed == 1.0 else slow_table(e.clip.table, speed)
    audio = slow_audio(e.clip.audio, speed)
    frames = renderer.render_frames(e.clip.robot, table, profile)
    if speech_strip:
        frames = overlay_speech_strip(frames, speech_envelope(audio, e.clip.sr, len(frames)))
    footnote = "" if speed == 1.0 else f"slow motion ({speed:g}x)"
    card = renderer.card_png(f"Clip {e.number}", e.clip.card_line, footnote)
    n_card = int(round(card_seconds * FPS))
    encode_frames([card] * n_card + frames, out_mp4, fps=FPS, audio=audio, sr=e.clip.sr,
                  audio_offset=card_seconds, work_dir=work_dir)
    return {"frames": len(frames), "seconds": (n_card + len(frames)) / FPS, "bytes": os.path.getsize(out_mp4),
            "speed": speed}


def render_gap(renderer: ViewerRenderer, out_mp4: str, seconds: float = GAP_SECONDS) -> str:
    black = renderer.black_png()
    return encode_frames([black] * int(round(seconds * FPS)), out_mp4, fps=FPS)


def render_reels(plans: Dict[str, List[Reel]], renderer: ViewerRenderer, run_dir: str, card_seconds: float = CARD_SECONDS,
                 gap_seconds: float = GAP_SECONDS, speech_strip: bool = True, speed: float = 1.0,
                 log=print) -> Dict[str, List[Reel]]:
    """Render every numbered clip and concatenate each reel. Fills ``Reel.path``/``render``."""
    clips_dir = os.path.join(run_dir, "reels", "parts")
    reels_dir = os.path.join(run_dir, "reels")
    os.makedirs(clips_dir, exist_ok=True)
    gap = os.path.join(clips_dir, "gap.mp4")
    render_gap(renderer, gap, gap_seconds)
    profiles = {robot: find_robot(robot) for robot in plans}
    for robot, reels in plans.items():
        for r in reels:
            parts: List[str] = []
            info: Dict[str, Dict] = {}
            for e in r.entries:
                p = os.path.join(clips_dir, f"{robot}_clip{e.number:03d}.mp4")
                info[str(e.number)] = render_entry(renderer, e, profiles[robot], p, card_seconds, speech_strip, speed=speed)
                log(f"[render] {robot} clip {e.number:3d} <- {e.clip.id}: {info[str(e.number)]['frames']} frames")
                parts.extend([p, gap])
            r.path = os.path.join(reels_dir, f"{r.name}.mp4")
            concat_mp4(parts, r.path)
            r.render = {"parts": info, "bytes": os.path.getsize(r.path)}
            log(f"[reel] {r.name}: clips {r.numbers[0]}..{r.numbers[-1]} ({len(r.entries)}), {r.seconds:.1f}s, "
                f"{r.render['bytes'] / 1e6:.1f} MB")
    return plans
