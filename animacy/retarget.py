"""Canonical human motion → robot joints, driven entirely by ``ROBOT.md``.

Two paths share one arithmetic core:

* :func:`retarget_clip` — offline. Time is *stretched* wherever the mapping
  would demand more than ``max_speed`` (the Autonomous OS playback rule), then
  resampled onto the robot's rate grid and smoothed with a zero-phase filter
  (or, for joints that declare a ``spring``, tracked by the same causal
  spring the live path uses, fed a lag-compensated input). Nothing is clipped,
  so a big human move becomes a slower robot move rather than a truncated one.
* :class:`LiveRetargeter` — online, one frame in, one frame out. Velocity is
  clipped (you cannot stretch time you have not seen yet). The browser viewer
  implements the same equations in JS (``web/js/retarget.js``).

Per joint and frame (exact spec with a numeric example in ``docs/RETARGET.md``):

``u = rest + offset + Σ gain·channel`` → deadband → soft limit → clamp
→ + gated idle sway → clamp → tracker (spring | one-pole) → rate limit → clamp.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .profile import Idle, Mapping, Profile, Spring
from .schema import HumanClip

# ---------------------------------------------------------------- constants
# Idle sway generator (docs/RETARGET.md §idle): three sines around `hz`.
IDLE_RATIOS = (1.0, 1.31, 0.67)          # frequency multipliers
IDLE_WEIGHTS = (0.5, 0.3, 0.2)           # sum = 1 → peak amplitude ≤ amp
IDLE_GOLDEN = 2.39996322972865332        # golden angle, radians
IDLE_RELEASE_S = 0.5                     # activity envelope release time constant


def idle_phase(joint_index: int, k: int) -> float:
    """Fixed phase of sine ``k`` (0..2) for the joint at ``joint_index`` in ``profile.joints``."""
    return math.fmod(IDLE_GOLDEN * (3 * joint_index + k + 1), 2.0 * math.pi)


def idle_value(t: float, amp: float, hz: float, joint_index: int) -> float:
    """Band-limited sway at time ``t`` (seconds on the joint's idle clock)."""
    s = 0.0
    for k in range(3):
        s += IDLE_WEIGHTS[k] * math.sin(2.0 * math.pi * IDLE_RATIOS[k] * hz * t + idle_phase(joint_index, k))
    return amp * s


def soft_clip(x, lo: float, hi: float, frac: Optional[float]):
    """tanh knee over the last ``frac`` of the range at each end (C1 continuous,
    identity in the middle, asymptotes to the bound). ``frac`` None/0 → identity.
    Works on floats and numpy arrays."""
    if not frac:
        return x
    k = frac * (hi - lo)
    if k <= 0:
        return x
    top, bot = hi - k, lo + k
    if isinstance(x, np.ndarray):
        y = x.copy()
        hi_m = y > top
        lo_m = y < bot
        y[hi_m] = top + k * np.tanh((y[hi_m] - top) / k)
        y[lo_m] = bot - k * np.tanh((bot - y[lo_m]) / k)
        return y
    if x > top:
        return top + k * math.tanh((x - top) / k)
    if x < bot:
        return bot - k * math.tanh((bot - x) / k)
    return x


def spring_step(y: float, v: float, u: float, dt: float, hz: float, zeta: float) -> Tuple[float, float]:
    """One semi-implicit Euler step of ``y'' = w²(u − y) − 2ζw y'``. Returns (y, v)."""
    w = 2.0 * math.pi * hz
    acc = w * w * (u - y) - 2.0 * zeta * w * v
    v = v + acc * dt
    y = y + v * dt
    return y, v


def spring_lag_s(spring: Spring) -> float:
    """Low-frequency group delay of the spring tracker, 2ζ/ω = ζ/(π·hz) seconds."""
    return spring.zeta / (math.pi * spring.hz)


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


def mapping_bounds(j, m: Optional[Mapping]) -> Tuple[float, float]:
    if m is None:
        return j.min, j.max
    return (j.min if m.min is None else m.min), (j.max if m.max is None else m.max)


def raw_joint_targets(frames: pd.DataFrame, profile: Profile, mode: str = "default") -> pd.DataFrame:
    """Mapping arithmetic only (no idle, no tracker, no speed limit): mix →
    deadband → offset+rest → soft limit → clamp. Joints without a mapping in
    ``mode`` sit at ``rest``."""
    mp = profile.mapping(mode)
    out = {"t": frames["t"].to_numpy(dtype=np.float64)}
    for j in profile.joints:
        m = mp.get(j.name)
        if m is None:
            out[j.name] = np.full(len(frames), j.rest, dtype=np.float64)
            continue
        x = _deadband(_mix(frames, m), m.deadband) + m.offset + j.rest
        lo, hi = mapping_bounds(j, m)
        out[j.name] = np.clip(soft_clip(x, lo, hi, m.soft_limit), lo, hi)
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


def idle_offline(u: np.ndarray, dt: float, idle: Idle, joint_index: int, lo: float, hi: float) -> np.ndarray:
    """Add the gated idle sway to an on-grid target series (same per-frame rule as live)."""
    out = u.copy()
    e = 0.0
    p = u[0]
    t = 0.0
    still = idle.still_speed()
    decay = math.exp(-dt / IDLE_RELEASE_S)
    for i in range(len(u)):
        a = abs(u[i] - p) / dt
        p = u[i]
        e = max(a, e * decay)
        g = min(max(1.0 - e / still, 0.0), 1.0)
        t += dt
        out[i] = min(max(u[i] + g * idle_value(t, idle.amp, idle.hz, joint_index), lo), hi)
    return out


def _advance(x: np.ndarray, frames: float) -> np.ndarray:
    """``x`` shifted earlier by ``frames`` (fractional, linear interpolation, end held)."""
    if frames <= 0 or len(x) < 2:
        return x
    n = len(x)
    src = np.arange(n) + frames
    return np.interp(np.clip(src, 0, n - 1), np.arange(n), x)


def spring_track_offline(u: np.ndarray, dt: float, spring: Spring, max_speed: float, lo: float, hi: float) -> np.ndarray:
    """Causal spring over an on-grid target, input advanced by the spring's
    low-frequency lag so the output lines up with the audio clock. Rate limit
    and clamp inside the loop, velocity re-derived after them (as live)."""
    adv = _advance(u, spring_lag_s(spring) / dt)
    out = np.empty_like(u)
    y, v = float(u[0]), 0.0
    vmax = max_speed * dt
    for i in range(len(u)):
        yn, v = spring_step(y, v, float(adv[i]), dt, spring.hz, spring.zeta)
        yn = y + min(max(yn - y, -vmax), vmax)
        yn = min(max(yn, lo), hi)
        v = (yn - y) / dt
        y = yn
        out[i] = y
    return out


def retarget_clip(clip: HumanClip, profile: Profile, mode: str = "default",
                  default_smooth_hz: Optional[float] = 8.0) -> pd.DataFrame:
    """Offline retarget. Returns ``t`` + one column per joint, on the robot's
    ``rate_hz`` grid, speed-legal for every joint, smoothed."""
    frames = clip.frames
    raw = raw_joint_targets(frames, profile, mode)
    t_stretched = stretch_timeline(raw["t"].to_numpy(), raw, profile)
    on_grid = resample(t_stretched, raw, profile.rate_hz)
    mp = profile.mapping(mode)
    dt = 1.0 / profile.rate_hz
    for idx, j in enumerate(profile.joints):
        m = mp.get(j.name)
        lo, hi = mapping_bounds(j, m)
        u = on_grid[j.name].to_numpy(dtype=np.float64)
        if m is not None and m.idle is not None:
            u = idle_offline(u, dt, m.idle, idx, lo, hi)
        if m is not None and m.spring is not None:
            v = spring_track_offline(u, dt, m.spring, j.max_speed, j.min, j.max)
        else:
            cutoff = default_smooth_hz if (m is None or m.smooth_hz is None) else m.smooth_hz
            v = smooth_offline(u, profile.rate_hz, cutoff)
            v = rate_limit(v, j.max_speed, profile.rate_hz)
        on_grid[j.name] = np.clip(v, j.min, j.max)
    return on_grid


class LiveRetargeter:
    """Streaming retarget: one frame in, one frame out (docs/RETARGET.md).

    >>> rt = LiveRetargeter(profile, mode="default")
    >>> joints = rt.step({"head_yaw": 12.0, "head_pitch": -3.0, ...}, dt=1/30)

    ``state`` holds the last output per joint; ``vel``, ``env``, ``prev_target``
    and ``clock`` are the spring velocity, idle activity envelope, previous
    pre-idle target and idle clock — all reset by :meth:`reset`.
    """

    def __init__(self, profile: Profile, mode: str = "default", default_smooth_hz: float = 6.0):
        self.profile = profile
        self.mode = mode
        self.mp = profile.mapping(mode)
        self.default_smooth_hz = default_smooth_hz
        self.reset()

    def reset(self) -> None:
        self.state: Dict[str, float] = {j.name: j.rest for j in self.profile.joints}
        self.vel: Dict[str, float] = {j.name: 0.0 for j in self.profile.joints}
        self.env: Dict[str, float] = {j.name: 0.0 for j in self.profile.joints}
        self.prev_target: Dict[str, float] = {j.name: j.rest for j in self.profile.joints}
        self.clock: Dict[str, float] = {j.name: 0.0 for j in self.profile.joints}

    def step(self, channels: Dict[str, float], dt: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for idx, j in enumerate(self.profile.joints):
            m = self.mp.get(j.name)
            lo, hi = mapping_bounds(j, m)
            if m is None:
                u = j.rest
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
                u = x + m.offset + j.rest
                u = soft_clip(u, lo, hi, m.soft_limit)
                u = min(max(u, lo), hi)
                cutoff = self.default_smooth_hz if m.smooth_hz is None else m.smooth_hz
                if m.idle is not None:
                    a = abs(u - self.prev_target[j.name]) / dt
                    self.prev_target[j.name] = u
                    e = max(a, self.env[j.name] * math.exp(-dt / IDLE_RELEASE_S))
                    self.env[j.name] = e
                    g = min(max(1.0 - e / m.idle.still_speed(), 0.0), 1.0)
                    self.clock[j.name] += dt
                    u = min(max(u + g * idle_value(self.clock[j.name], m.idle.amp, m.idle.hz, idx), lo), hi)
            prev = self.state[j.name]
            if m is not None and m.spring is not None:
                y, v = spring_step(prev, self.vel[j.name], u, dt, m.spring.hz, m.spring.zeta)
            else:
                # one-pole low-pass, alpha from cutoff
                alpha = 1.0 if not cutoff else 1.0 - math.exp(-2.0 * math.pi * cutoff * dt)
                y = prev + alpha * (u - prev)
            # velocity clip
            vmax = j.max_speed * dt
            y = prev + min(max(y - prev, -vmax), vmax)
            y = min(max(y, j.min), j.max)
            self.vel[j.name] = (y - prev) / dt
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
