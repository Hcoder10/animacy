"""Canonical human motion space (``animacy.human.v1``).

This module is the single source of truth for channel names, units and file
layout. Everything else — capture, retarget, model, web export — imports
``CHANNELS`` from here rather than spelling column names out.

See ``docs/CANONICAL.md`` for the meaning and sign of every channel.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCHEMA = "animacy.human.v1"
RATE_HZ = 30.0
AUDIO_SR = 16000

# (name, unit, lo, hi) — lo/hi are sanity bounds, not hard clamps.
_SPEC = [
    ("t", "s", 0.0, np.inf),
    ("head_yaw", "deg", -90, 90),
    ("head_pitch", "deg", -60, 60),
    ("head_roll", "deg", -45, 45),
    ("head_x", "mm", -150, 150),
    ("head_y", "mm", -150, 150),
    ("head_z", "mm", -150, 150),
    ("gaze_yaw", "deg", -40, 40),
    ("gaze_pitch", "deg", -30, 30),
    ("brow_l", "unit", 0, 1),
    ("brow_r", "unit", 0, 1),
    ("brow_furrow", "unit", 0, 1),
    ("eye_open_l", "unit", 0, 1),
    ("eye_open_r", "unit", 0, 1),
    ("mouth_open", "unit", 0, 1),
    ("smile", "unit", 0, 1),
    ("torso_lean_fwd", "deg", -45, 45),
    ("torso_lean_side", "deg", -45, 45),
    ("torso_yaw", "deg", -90, 90),
    ("arm_valid", "flag", 0, 1),
    ("shoulder_yaw", "deg", -90, 90),
    ("shoulder_pitch", "deg", -30, 180),
    ("elbow_flex", "deg", 0, 150),
    ("wrist_roll", "deg", -90, 90),
    ("wrist_pitch", "deg", -80, 80),
    ("hand_open", "unit", 0, 1),
    ("speaking", "flag", 0, 1),
    ("face_valid", "flag", 0, 1),
]

CHANNELS: List[str] = [s[0] for s in _SPEC]
UNITS: Dict[str, str] = {s[0]: s[1] for s in _SPEC}
BOUNDS: Dict[str, tuple] = {s[0]: (s[2], s[3]) for s in _SPEC}
FLAGS = [c for c in CHANNELS if UNITS[c] == "flag"]
# Channels a retarget mapping may reference (everything but time and flags).
MAPPABLE: List[str] = [c for c in CHANNELS if c != "t" and UNITS[c] != "flag"]
FACE_CHANNELS = [
    "head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z",
    "gaze_yaw", "gaze_pitch", "brow_l", "brow_r", "brow_furrow",
    "eye_open_l", "eye_open_r", "mouth_open", "smile",
]
ARM_CHANNELS = ["shoulder_yaw", "shoulder_pitch", "elbow_flex", "wrist_roll", "wrist_pitch", "hand_open"]
TORSO_CHANNELS = ["torso_lean_fwd", "torso_lean_side", "torso_yaw"]

MOTION_FILE = "motion.parquet"
AUDIO_FILE = "audio.wav"
META_FILE = "meta.json"


def empty_frames(n: int, rate_hz: float = RATE_HZ) -> pd.DataFrame:
    """A neutral clip of ``n`` frames: zeros everywhere, all validity flags 0."""
    df = pd.DataFrame({c: np.zeros(n, dtype=np.float32) for c in CHANNELS})
    df["t"] = (np.arange(n) / rate_hz).astype(np.float32)
    # A face at rest has half-open eyes; keep neutral meaningful, not zero.
    df["eye_open_l"] = 0.6
    df["eye_open_r"] = 0.6
    return df


@dataclass
class HumanClip:
    """One captured or generated clip in the canonical space."""

    frames: pd.DataFrame
    meta: Dict = field(default_factory=dict)
    audio: Optional[np.ndarray] = None  # float32 mono at ``sr``
    sr: int = AUDIO_SR

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_frames(cls, frames: pd.DataFrame, **meta) -> "HumanClip":
        clip = cls(frames=frames.copy(), meta=dict(meta))
        clip.meta.setdefault("schema", SCHEMA)
        clip.meta.setdefault("rate_hz", RATE_HZ)
        clip._normalise_columns()
        return clip

    def _normalise_columns(self) -> None:
        for c in CHANNELS:
            if c not in self.frames.columns:
                self.frames[c] = 0.0 if c in FLAGS or c == "t" else np.nan
        self.frames = self.frames[CHANNELS].astype(np.float32)

    # ---- properties ---------------------------------------------------------
    @property
    def rate_hz(self) -> float:
        return float(self.meta.get("rate_hz", RATE_HZ))

    @property
    def duration(self) -> float:
        return float(self.frames["t"].iloc[-1]) if len(self.frames) else 0.0

    def __len__(self) -> int:
        return len(self.frames)

    # ---- validation ---------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of problems; empty means the clip is well-formed."""
        errs: List[str] = []
        f = self.frames
        missing = [c for c in CHANNELS if c not in f.columns]
        if missing:
            errs.append(f"missing channels: {missing}")
            return errs
        t = f["t"].to_numpy()
        if len(t) and not np.all(np.diff(t) > 0):
            errs.append("t is not strictly increasing")
        for c in FLAGS:
            v = f[c].to_numpy()
            if not np.all(np.isin(v[~np.isnan(v)], [0, 1])):
                errs.append(f"{c} must be 0/1")
        face_ok = f["face_valid"].to_numpy() > 0
        for c in FACE_CHANNELS:
            v = f[c].to_numpy()
            if np.any(np.isnan(v[face_ok])):
                errs.append(f"{c} is NaN on frames where face_valid=1")
                break
        arm_ok = f["arm_valid"].to_numpy() > 0
        for c in ARM_CHANNELS:
            v = f[c].to_numpy()
            if np.any(np.isnan(v[arm_ok])):
                errs.append(f"{c} is NaN on frames where arm_valid=1")
                break
        for c, (lo, hi) in BOUNDS.items():
            v = f[c].to_numpy()
            v = v[~np.isnan(v)]
            if len(v) and (v.min() < lo - 1e-6 or v.max() > hi + 1e-6):
                errs.append(f"{c} outside sanity bounds [{lo}, {hi}]: min={v.min():.2f} max={v.max():.2f}")
        if self.audio is not None and len(self.audio):
            adur = len(self.audio) / self.sr
            if abs(adur - self.duration) > 0.5:
                errs.append(f"audio ({adur:.2f}s) and motion ({self.duration:.2f}s) differ by >0.5s")
        return errs

    # ---- io -----------------------------------------------------------------
    def save(self, clip_dir: str) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq

        os.makedirs(clip_dir, exist_ok=True)
        table = pa.Table.from_pandas(self.frames[CHANNELS], preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"animacy": json.dumps(self.meta).encode()})
        pq.write_table(table, os.path.join(clip_dir, MOTION_FILE))
        with open(os.path.join(clip_dir, META_FILE), "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2)
        if self.audio is not None:
            import soundfile as sf

            sf.write(os.path.join(clip_dir, AUDIO_FILE), self.audio.astype(np.float32), self.sr)
        return clip_dir

    @classmethod
    def load(cls, clip_dir: str, audio: bool = True) -> "HumanClip":
        import pyarrow.parquet as pq

        frames = pq.read_table(os.path.join(clip_dir, MOTION_FILE)).to_pandas()
        meta_path = os.path.join(clip_dir, META_FILE)
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        clip = cls.from_frames(frames, **meta)
        wav = os.path.join(clip_dir, AUDIO_FILE)
        if audio and os.path.exists(wav):
            import soundfile as sf

            data, sr = sf.read(wav, dtype="float32", always_2d=True)
            clip.audio, clip.sr = data.mean(axis=1), sr
        return clip

    def to_web_json(self, channels: Optional[List[str]] = None, decimals: int = 3) -> Dict:
        """Compact JSON for the browser viewer: column-major float arrays."""
        cols = channels or CHANNELS
        out = {"schema": SCHEMA, "rate_hz": self.rate_hz, "n": len(self.frames), "channels": cols}
        arr = self.frames[cols].to_numpy(dtype=np.float64)
        arr = np.where(np.isnan(arr), None, np.round(arr, decimals))
        out["data"] = {c: arr[:, i].tolist() for i, c in enumerate(cols)}
        return out
