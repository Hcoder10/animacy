"""Data → numbers for ``ROBOT.md`` mappings (no runtime code lives here).

* :func:`joint_stats` — per-joint amplitude / velocity / stillness statistics
  of any joint table set (the vendor's native clips, or human clips retargeted
  through a mapping).
* :func:`propose_multipliers` — per-joint gain multipliers that make the
  retargeted p95 excursion match the vendor envelope, capped by the joint's
  headroom and by ``[min_mult, max_mult]``.
* :func:`gaze_jacobian` / :func:`gaze_compensation` — linearised URDF FK of
  the head's pointing direction, turned into extra ``mix`` terms on the gaze
  joint so leaning/rising does not drag the gaze off the person.
* :func:`rewrite_gains` — writes fitted gains back into the ``retarget:`` block
  of a ROBOT.md as text (comments and layout preserved), stamping each changed
  line with ``# fitted by scripts/retarget_fit.py <date>``.

``scripts/retarget_fit.py`` and ``scripts/retarget_eval.py`` are the CLIs.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .profile import Profile
from .retarget import _deadband, _mix, raw_joint_targets, retarget_clip
from .schema import HumanClip

# Human clips whose tracking is too broken to fit against (multi-shot b-roll,
# 400x300 handheld) — the same two `checkpoints/v1/REPORT.md` excludes.
DEFAULT_EXCLUDE = ("sd_rapper_interview", "cbp_vlog_day2")
JUNK_MM = 140.0          # |head_x/y/z| at the ±150 sanity clamp = tracker failure
STILL_UNITS_PER_S = 5.0  # RESULTS.md stillness threshold


# ---------------------------------------------------------------- statistics
@dataclass
class JointStats:
    n: int
    p5: float
    p50: float
    p95: float
    abs_p95: float
    abs_p99: float
    vel_p50: float
    vel_p95: float
    vel_p99: float
    vel_max: float
    still: float             # fraction of frames slower than STILL_UNITS_PER_S
    speeds: np.ndarray       # per-frame |dv/dt| (for histogram distances)
    values: np.ndarray       # centred values

    def row(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in ("n", "p5", "p50", "p95", "abs_p95", "abs_p99", "vel_p50", "vel_p95", "vel_p99", "vel_max", "still")}


def joint_stats(tables: Sequence[pd.DataFrame], joints: Sequence[str], center: Dict[str, float]) -> Dict[str, JointStats]:
    """Statistics of ``value − center[joint]`` and of |velocity| over all tables."""
    out: Dict[str, JointStats] = {}
    for j in joints:
        vals, spd = [], []
        for tb in tables:
            if j not in tb.columns or len(tb) < 2:
                continue
            v = tb[j].to_numpy(dtype=np.float64) - center.get(j, 0.0)
            t = tb["t"].to_numpy(dtype=np.float64)
            dt = np.maximum(np.diff(t), 1e-3)
            vals.append(v)
            spd.append(np.abs(np.diff(v)) / dt)
        if not vals:
            continue
        v = np.concatenate(vals)
        s = np.concatenate(spd) if spd else np.zeros(0)
        out[j] = JointStats(
            n=len(v), p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
            abs_p95=float(np.percentile(np.abs(v), 95)), abs_p99=float(np.percentile(np.abs(v), 99)),
            vel_p50=float(np.percentile(s, 50)) if len(s) else 0.0, vel_p95=float(np.percentile(s, 95)) if len(s) else 0.0,
            vel_p99=float(np.percentile(s, 99)) if len(s) else 0.0, vel_max=float(s.max()) if len(s) else 0.0,
            still=float(np.mean(s < STILL_UNITS_PER_S)) if len(s) else 1.0, speeds=s, values=v)
    return out


def library_center(tables: Sequence[pd.DataFrame], joints: Sequence[str]) -> Dict[str, float]:
    """Library-wide median per joint: where the vendor's clips actually live
    (can differ from ROBOT.md ``rest`` — the lamp's wrist_pitch does)."""
    out = {}
    for j in joints:
        v = np.concatenate([tb[j].to_numpy(dtype=np.float64) for tb in tables if j in tb.columns])
        out[j] = float(np.median(v))
    return out


def velocity_w1(a: JointStats, b: JointStats) -> float:
    """Wasserstein-1 between two |velocity| distributions, relative to ``b``'s mean speed."""
    from scipy.stats import wasserstein_distance

    if len(a.speeds) == 0 or len(b.speeds) == 0:
        return float("nan")
    scale = max(float(np.mean(b.speeds)), 1e-6)
    return float(wasserstein_distance(a.speeds, b.speeds) / scale)


# ---------------------------------------------------------------- clip loading
def load_native_clips(profile: Profile, subset: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """The vendor's own clips as joint tables (``t`` + joint columns)."""
    from .export import read_autonomous_os_csv

    if profile.native_clips is None:
        return {}
    d = os.path.join(profile.dir, profile.native_clips.dir)
    want = set(subset) if subset is not None else None
    out: Dict[str, pd.DataFrame] = {}
    for fn in sorted(os.listdir(d)):
        name, ext = os.path.splitext(fn)
        if name == "index" or (want is not None and name not in want):
            continue
        if profile.native_clips.format == "autonomous_os_csv" and ext == ".csv":
            out[name] = read_autonomous_os_csv(os.path.join(d, fn))
        elif profile.native_clips.format == "json" and ext == ".json":
            j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            tb = {"t": np.asarray(j["t"], dtype=np.float64)}
            for k, v in j["data"].items():
                tb[k] = np.asarray(v, dtype=np.float64)
            out[name] = pd.DataFrame(tb)
    return out


def load_human_clips(root: str, names: Optional[Sequence[str]] = None, exclude: Sequence[str] = DEFAULT_EXCLUDE,
                     clean: bool = True) -> Dict[str, HumanClip]:
    """Canonical clips under ``root``; ``clean`` keeps only face-valid frames
    with sane head translation (tracker failures clamp at ±150 mm)."""
    out: Dict[str, HumanClip] = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "motion.parquet")):
            continue
        if names is not None and d not in names:
            continue
        if names is None and d in exclude:
            continue
        clip = HumanClip.load(p, audio=False)
        if clean:
            clip = clean_clip(clip)
        if len(clip.frames) > 30:
            out[d] = clip
    return out


def clean_clip(clip: HumanClip) -> HumanClip:
    f = clip.frames
    ok = f["face_valid"].to_numpy() > 0
    for c in ("head_x", "head_y", "head_z"):
        v = f[c].to_numpy()
        ok &= ~(np.abs(np.nan_to_num(v, nan=0.0)) >= JUNK_MM)
    kept = f[ok].reset_index(drop=True).copy()
    # keep a real clock: contiguous 1/rate steps so velocities stay meaningful
    kept["t"] = np.arange(len(kept)) / clip.rate_hz
    c = HumanClip(frames=kept, meta=dict(clip.meta, cleaned=True, kept_fraction=float(ok.mean())))
    c._normalise_columns()
    return c


# ---------------------------------------------------------------- retarget stats
def unclamped_targets(clip: HumanClip, profile: Profile, mode: str = "default",
                      exclude: Optional[Dict[str, Sequence[str]]] = None) -> pd.DataFrame:
    """Rest-relative mapped values with deadband but WITHOUT soft limit / clamp:
    linear in the gains, which is what a gain fit needs. ``exclude[joint]``
    lists channels to leave out of that joint's mix (e.g. the gaze joint's
    compensation terms, which are derived, not fitted)."""
    mp = profile.mapping(mode)
    exclude = exclude or {}
    out = {"t": clip.frames["t"].to_numpy(dtype=np.float64)}
    for j in profile.joints:
        m = mp.get(j.name)
        if m is None:
            out[j.name] = np.zeros(len(clip.frames))
            continue
        skip = set(exclude.get(j.name, ()))
        x = np.zeros(len(clip.frames), dtype=np.float64)
        for term in m.terms():
            if term.from_ in skip:
                continue
            x += term.gain * np.nan_to_num(clip.frames[term.from_].to_numpy(dtype=np.float64), nan=0.0)
        out[j.name] = _deadband(x, m.deadband) + m.offset
    return pd.DataFrame(out)


def retarget_tables(clips: Dict[str, HumanClip], profile: Profile, mode: str = "default") -> Dict[str, pd.DataFrame]:
    return {n: retarget_clip(c, profile, mode) for n, c in clips.items()}


# ---------------------------------------------------------------- gain proposals
@dataclass
class Proposal:
    joint: str
    vendor_abs_p95: float
    current_abs_p95: float
    raw_mult: float
    mult: float
    cap_reason: str


def propose_multipliers(profile: Profile, mode: str, vendor: Dict[str, JointStats], current: Dict[str, JointStats],
                        min_mult: float = 0.5, max_mult: float = 3.0, headroom: float = 0.9,
                        skip: Sequence[str] = (), pipeline: Optional[Dict[str, JointStats]] = None,
                        vel_cap: Optional[float] = 1.25) -> Dict[str, Proposal]:
    """Multiplier per mapped joint so that ``current.abs_p95 * mult == vendor.abs_p95``,
    clipped to ``[min_mult, max_mult]``, to the headroom of the mapping bounds
    (scaled p99 on each side must stay inside ``headroom × distance to bound``)
    and, when ``pipeline`` (full-pipeline statistics of the same clips) is
    given, so that the scaled velocity p95 stays within ``vel_cap × vendor.vel_p95``
    — human heads are brisker than authored robot clips, and amplitude alone
    would make the body whip. ``current`` must be statistics of
    :func:`unclamped_targets` (linear in the gains)."""
    mp = profile.mapping(mode)
    out: Dict[str, Proposal] = {}
    for j in profile.joints:
        m = mp.get(j.name)
        if m is None or j.name in skip or j.name not in vendor or j.name not in current:
            continue
        v, c = vendor[j.name], current[j.name]
        if c.abs_p95 < 1e-9 or v.abs_p95 < 1e-9:
            continue
        raw = v.abs_p95 / c.abs_p95
        mult, reason = raw, "fit"
        if mult > max_mult:
            mult, reason = max_mult, f"capped at max_mult {max_mult}"
        if pipeline is not None and vel_cap and j.name in pipeline and pipeline[j.name].vel_p95 > 1e-9:
            kv = vel_cap * v.vel_p95 / pipeline[j.name].vel_p95
            if kv < mult:
                mult, reason = kv, f"velocity cap ({vel_cap}x vendor vel p95 {v.vel_p95:.0f})"
        if mult < min_mult:
            mult, reason = min_mult, f"raised to min_mult {min_mult}"
        lo = j.min if m.min is None else m.min
        hi = j.max if m.max is None else m.max
        pos99 = float(np.percentile(c.values, 99))
        neg01 = float(np.percentile(c.values, 1))
        if pos99 > 1e-9:
            cap = headroom * (hi - j.rest - m.offset) / pos99
            if cap < mult:
                mult, reason = max(cap, min_mult), f"headroom to max {hi}"
        if neg01 < -1e-9:
            cap = headroom * (j.rest + m.offset - lo) / (-neg01)
            if cap < mult:
                mult, reason = max(cap, min_mult), f"headroom to min {lo}"
        out[j.name] = Proposal(j.name, v.abs_p95, c.abs_p95, raw, mult, reason)
    return out


# ---------------------------------------------------------------- gaze (URDF FK)
LOOK_LOCAL = np.array([0.70, 0.0, -0.71]) / np.linalg.norm([0.70, 0.0, -0.71])  # lamp head: look axis in the head frame


def load_urdf(profile: Profile):
    import yourdfpy

    return yourdfpy.URDF.load(profile.urdf_path(), load_meshes=False, build_scene_graph=True)


def head_pointing(urdf, profile: Profile, q: Dict[str, float], head_link: str = "head",
                  look_local: np.ndarray = LOOK_LOCAL) -> Tuple[float, float, np.ndarray]:
    """(elevation deg, azimuth deg, position m) of the head's look axis in the
    base frame, for joint values ``q`` in profile units."""
    cfg = {}
    for j in profile.joints:
        v = (q.get(j.name, j.rest) + j.urdf_offset) * j.urdf_sign
        cfg[j.urdf_joint] = math.radians(v) if j.unit == "deg" else v / 1000.0 if j.unit == "mm" else v
    urdf.update_cfg(cfg)
    M = urdf.get_transform(head_link, urdf.base_link)
    look = M[:3, :3] @ look_local
    return math.degrees(math.asin(max(-1.0, min(1.0, look[2])))), math.degrees(math.atan2(look[1], look[0])), M[:3, 3]


def gaze_jacobian(urdf, profile: Profile, joints: Sequence[str], h: float = 1.0) -> Dict[str, Tuple[float, float]]:
    """Central-difference d(elevation, azimuth)/d(joint) at rest, deg per unit."""
    rest = {j.name: j.rest for j in profile.joints}
    out = {}
    for jn in joints:
        ep, ap, _ = head_pointing(urdf, profile, dict(rest, **{jn: rest[jn] + h}))
        em, am, _ = head_pointing(urdf, profile, dict(rest, **{jn: rest[jn] - h}))
        out[jn] = ((ep - em) / (2 * h), (ap - am) / (2 * h))
    return out


def gaze_compensation(profile: Profile, mode: str, jac: Dict[str, Tuple[float, float]], gaze_joint: str,
                      body_joints: Sequence[str]) -> Dict[str, float]:
    """For every channel feeding a body joint, the gain to add on ``gaze_joint``
    so the elevation change cancels: ``g = −Σ_b J_b·g_b / J_gaze``."""
    mp = profile.mapping(mode)
    jg = jac[gaze_joint][0]
    comp: Dict[str, float] = {}
    for b in body_joints:
        m = mp.get(b)
        if m is None:
            continue
        for term in m.terms():
            comp[term.from_] = comp.get(term.from_, 0.0) - jac[b][0] * term.gain / jg
    return comp


def gaze_error_cases(profile: Profile, mode: str, urdf, cases: Sequence[Dict[str, float]], gaze_joint: str = "wrist_pitch") -> List[Tuple[Dict[str, float], float, float]]:
    """(channels, elevation error deg, azimuth error deg) for each channel pose,
    evaluated through the mapping arithmetic and the URDF FK."""
    from .schema import empty_frames

    rest = {j.name: j.rest for j in profile.joints}
    e0, a0, _ = head_pointing(urdf, profile, rest)
    out = []
    for ch in cases:
        f = empty_frames(1)
        for k, v in ch.items():
            f[k] = v
        q = raw_joint_targets(f, profile, mode).iloc[0].to_dict()
        q.pop("t", None)
        e, a, _ = head_pointing(urdf, profile, q)
        out.append((ch, e - e0, a - a0))
    return out


# ---------------------------------------------------------------- ROBOT.md text surgery
_TERM_RE = re.compile(r"^(?P<pre>\s*-\s*\{\s*from:\s*(?P<ch>[a-z_]+))(?P<gain>\s*,\s*gain:\s*[-+0-9.eE]+)?(?P<post>\s*[,}].*)$")
_SINGLE_RE = re.compile(r"^(?P<pre>\s*(?P<joint>[a-z_0-9]+):\s*\{\s*from:\s*(?P<ch>[a-z_]+))(?P<gain>\s*,\s*gain:\s*[-+0-9.eE]+)?(?P<post>\s*[,}].*)$")
_STAMP_RE = re.compile(r"\s*#\s*fitted by scripts/retarget_fit\.py[^\n]*$")


def _fmt(g: float) -> str:
    s = f"{g:.4g}"
    if "e" in s or "E" in s:
        s = f"{g:.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def _gain_text(existing: Optional[str], g: float) -> str:
    """``, gain: <g>`` keeping the original spacing when the term already had a gain."""
    if existing:
        return re.sub(r"[-+0-9.eE]+$", _fmt(g), existing)
    return f", gain: {_fmt(g)}"


def _stamp(line: str, stamp: str) -> str:
    line = _STAMP_RE.sub("", line.rstrip("\n"))
    if stamp:
        # keep an existing plain comment in front of the stamp
        return f"{line}  # fitted by scripts/retarget_fit.py {stamp}"
    return line


def rewrite_gains(text: str, mode: str, gains: Dict[str, Dict[str, float]], stamp: str) -> str:
    """Set ``gains[joint][channel]`` in the ``retarget.<mode>`` block of a
    ROBOT.md text; every other byte is preserved. Terms that do not exist are
    an error (the fitter never invents structure — add the term by hand first)."""
    lines = text.split("\n")
    out = []
    in_front, in_retarget, cur_mode, cur_joint = True, False, None, None
    retarget_indent = mode_indent = joint_indent = None
    pending = {j: dict(ch) for j, ch in gains.items()}
    dashes = 0
    for line in lines:
        if line.strip() == "---":
            dashes += 1
            in_front = dashes < 2
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if in_front and stripped and not stripped.startswith("#"):
            key = stripped.split(":")[0] if ":" in stripped and not stripped.startswith("-") else None
            if indent == 0:
                in_retarget = key == "retarget"
                retarget_indent = 0
                cur_mode = cur_joint = None
            elif in_retarget:
                if key is not None and mode_indent is None and indent > retarget_indent and not stripped.startswith("-") and not stripped.startswith("{"):
                    mode_indent = indent
                if key is not None and indent == mode_indent:
                    cur_mode, cur_joint = key, None
                elif cur_mode == mode and key is not None and not stripped.startswith("-") and (joint_indent is None or indent == joint_indent) and indent > mode_indent:
                    joint_indent = indent
                    cur_joint = key
                    ms = _SINGLE_RE.match(line)
                    if ms and cur_joint in pending and ms.group("ch") in pending[cur_joint]:
                        g = pending[cur_joint].pop(ms.group("ch"))
                        line = _stamp(f"{ms.group('pre')}{_gain_text(ms.group('gain'), g)}{ms.group('post')}", stamp)
                elif cur_mode == mode and cur_joint in pending and stripped.startswith("-"):
                    mt = _TERM_RE.match(line)
                    if mt and mt.group("ch") in pending[cur_joint]:
                        g = pending[cur_joint].pop(mt.group("ch"))
                        line = _stamp(f"{mt.group('pre')}{_gain_text(mt.group('gain'), g)}{mt.group('post')}", stamp)
        out.append(line)
    left = {j: ch for j, ch in pending.items() if ch}
    if left:
        raise KeyError(f"terms not found in retarget.{mode}: {left}")
    return "\n".join(out)


def current_gains(profile: Profile, mode: str) -> Dict[str, Dict[str, float]]:
    return {jn: {t.from_: t.gain for t in m.terms()} for jn, m in profile.mapping(mode).items()}


def scaled_gains(profile: Profile, mode: str, mults: Dict[str, float], fixed: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
    """Every term gain × its joint's multiplier, except ``fixed[joint][channel]`` which are set verbatim."""
    fixed = fixed or {}
    out: Dict[str, Dict[str, float]] = {}
    for jn, m in profile.mapping(mode).items():
        k = mults.get(jn, 1.0)
        out[jn] = {}
        for t in m.terms():
            if jn in fixed and t.from_ in fixed[jn]:
                out[jn][t.from_] = fixed[jn][t.from_]
            else:
                out[jn][t.from_] = t.gain * k
    return out


# ---------------------------------------------------------------- tables
def fmt_table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(header))) + " |"  # noqa: E731
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])
