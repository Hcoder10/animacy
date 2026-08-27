"""Canonical human motion → robot joints, driven entirely by ``ROBOT.md``.

Two paths share one arithmetic core:

* :func:`retarget_clip` — offline. Time is *stretched* wherever the mapping
  would demand more than ``max_speed`` (the Autonomous OS playback rule), then
  resampled onto the robot's rate grid and smoothed with a zero-phase filter.
  Nothing is clipped, so a big human move becomes a slower robot move rather
  than a truncated one.
* :class:`LiveRetargeter` — online, one frame in, one frame out. Velocity is
  clipped (you cannot stretch time you have not seen yet) and smoothing is a
  one-pole low-pass. The browser viewer implements the same equations in JS.

``joint = rest + offset + Σ gain · channel`` → deadband → clamp → speed →
smooth. See ``docs/ROBOT_MD_SPEC.md`` rule 2.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .profile import Mapping, Profile
from .schema import HumanClip


def _mix(frames: pd.DataFrame, m: Mapping) -> np.ndarray:
    out = np.zeros(len(frames), dtype=np.float64)
    for term in m.terms():
        v = frames[term.from_].to_numpy(dtype=np.float64)
        out += term.gain * np.nan_to_num(v, nan=0.0)
    return out


def _deadband(x: np.ndarray, db: float) -> np.ndarray:
    if db <= 0:
        return x
    return np.where(np.abs(x) < db, 0.0, x - np.sign(x) * db)


def raw_joint_targets(frames: pd.DataFrame, profile: Profile, mode: str = "default") -> pd.DataFrame:
    """Mapping arithmetic only (no speed limit, no smoothing). Joints without a
    mapping in ``mode`` sit at ``rest``."""
    mp = profile.mapping(mode)
    out = {"t": frames["t"].to_numpy(dtype=np.float64)}
    for j in profile.joints:
        m = mp.get(j.name)
        if m is None:
            out[j.name] = np.full(len(frames), j.rest, dtype=np.float64)
            continue
        x = _deadband(_mix(frames, m), m.deadband) + m.offset + j.rest
        lo = j.min if m.min is None else m.min
        hi = j.max if m.max is None else m.max
        out[j.name] = np.clip(x, lo, hi)
    return pd.DataFrame(out)


def stretch_timeline(t: np.ndarray, joints: pd.DataFrame, profile: Profile, margin: float = 0.92) -> np.ndarray:
    """Widen every frame gap that would exceed a joint's ``max_speed``.

    Same rule as Autonomous OS ``recording_timing.stretch_timeline``: only the
    impossible segments grow, everything else keeps its authored timing.
    ``margin`` leaves headroom for the zero-phase smoothing that follows
    (measured overshoot 1-6% without it); :func:`rate_limit` then guarantees
    legality exactly.
    """
    out = np.empty_like(t)
    out[0] = t[0]
    cols = [(j.name, j.max_speed * margin) for j in profile.joints if j.name in joints.columns]
    vals = {n: joints[n].to_numpy() for n, _ in cols}
    for i in range(1, len(t)):
        authored = max(t[i] - t[i - 1], 1e-3)
        needed = 0.0
        for n, vmax in cols:
            needed = max(needed, abs(vals[n][i] - vals[n][i - 1]) / vmax)
        out[i] = out[i - 1] + max(authored, needed)
    return out


def rate_limit(x: np.ndarray, max_speed: float, rate_hz: float) -> np.ndarray:
    """Causal per-frame velocity clamp: the exact guarantee, applied last."""
    step = max_speed / rate_hz
    out = x.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > step:
            out[i] = out[i - 1] + step
        elif d < -step:
            out[i] = out[i - 1] - step
    return out


def resample(t_src: np.ndarray, table: pd.DataFrame, rate_hz: float) -> pd.DataFrame:
    dur = float(t_src[-1] - t_src[0])
    n = max(1, int(round(dur * rate_hz))) + 1
    t_new = t_src[0] + np.arange(n) / rate_hz
    t_new[-1] = min(t_new[-1], t_src[-1])
    out = {"t": t_new - t_src[0]}
    for c in table.columns:
        if c == "t":
            continue
        out[c] = np.interp(t_new, t_src, table[c].to_numpy(dtype=np.float64))
    return pd.DataFrame(out)


def smooth_offline(x: np.ndarray, rate_hz: float, cutoff_hz: Optional[float]) -> np.ndarray:
    """Zero-phase Butterworth (no lag). ``cutoff_hz`` None → untouched."""
    if not cutoff_hz or len(x) < 12:
        return x
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * rate_hz
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(2, wn)
    return filtfilt(b, a, x, padlen=min(9, len(x) - 1))


def retarget_clip(clip: HumanClip, profile: Profile, mode: str = "default",
                  default_smooth_hz: Optional[float] = 8.0) -> pd.DataFrame:
    """Offline retarget. Returns ``t`` + one column per joint, on the robot's
    ``rate_hz`` grid, speed-legal for every joint, smoothed."""
    frames = clip.frames
    raw = raw_joint_targets(frames, profile, mode)
    t_stretched = stretch_timeline(raw["t"].to_numpy(), raw, profile)
    on_grid = resample(t_stretched, raw, profile.rate_hz)
    mp = profile.mapping(mode)
    for j in profile.joints:
        m = mp.get(j.name)
        cutoff = default_smooth_hz if (m is None or m.smooth_hz is None) else m.smooth_hz
        v = smooth_offline(on_grid[j.name].to_numpy(), profile.rate_hz, cutoff)
        v = rate_limit(v, j.max_speed, profile.rate_hz)
        on_grid[j.name] = np.clip(v, j.min, j.max)
    return on_grid


class LiveRetargeter:
    """Streaming retarget with velocity clipping and one-pole smoothing.

    >>> rt = LiveRetargeter(profile, mode="default")
    >>> joints = rt.step({"head_yaw": 12.0, "head_pitch": -3.0, ...}, dt=1/30)
    """

    def __init__(self, profile: Profile, mode: str = "default", default_smooth_hz: float = 6.0):
        self.profile = profile
        self.mode = mode
        self.mp = profile.mapping(mode)
        self.default_smooth_hz = default_smooth_hz
        self.state: Dict[str, float] = {j.name: j.rest for j in profile.joints}

    def reset(self) -> None:
        self.state = {j.name: j.rest for j in self.profile.joints}

    def step(self, channels: Dict[str, float], dt: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for j in self.profile.joints:
            m = self.mp.get(j.name)
            if m is None:
                target = j.rest
                cutoff = self.default_smooth_hz
            else:
                x = 0.0
                for term in m.terms():
                    v = channels.get(term.from_, 0.0)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        v = 0.0
                    x += term.gain * float(v)
                if m.deadband > 0:
                    x = 0.0 if abs(x) < m.deadband else x - math.copysign(m.deadband, x)
                target = x + m.offset + j.rest
                lo = j.min if m.min is None else m.min
                hi = j.max if m.max is None else m.max
                target = min(max(target, lo), hi)
                cutoff = self.default_smooth_hz if m.smooth_hz is None else m.smooth_hz
            prev = self.state[j.name]
            # one-pole low-pass, alpha from cutoff
            alpha = 1.0 if not cutoff else 1.0 - math.exp(-2.0 * math.pi * cutoff * dt)
            y = prev + alpha * (target - prev)
            # velocity clip
            vmax = j.max_speed * dt
            y = prev + min(max(y - prev, -vmax), vmax)
            y = min(max(y, j.min), j.max)
            self.state[j.name] = y
            out[j.name] = y
        return out


def to_urdf_values(joints: pd.DataFrame, profile: Profile) -> Dict[str, np.ndarray]:
    """Joint table → URDF joint values in radians/metres (sign + offset applied)."""
    out = {}
    for j in profile.joints:
        v = (joints[j.name].to_numpy(dtype=np.float64) + j.urdf_offset) * j.urdf_sign
        if j.unit == "deg":
            v = np.deg2rad(v)
        elif j.unit == "mm":
            v = v / 1000.0
        out[j.urdf_joint] = v
    return out
