"""``ROBOT.md`` parser, validator and JSON exporter (``animacy.robot.v1``).

A robot profile is YAML front matter followed by prose. The front matter is
the machine contract (joints, retarget mappings, export/runtime hints); the
prose is for people and coding agents. See ``docs/ROBOT_MD_SPEC.md``.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .schema import MAPPABLE

SCHEMA = "animacy.robot.v1"
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
_JOINT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class Joint(BaseModel):
    name: str
    unit: str = "deg"
    min: float
    max: float
    rest: float = 0.0
    max_speed: float = Field(..., gt=0, description="unit per second")
    urdf_joint: Optional[str] = None
    urdf_sign: float = 1.0
    urdf_offset: float = 0.0

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _JOINT_NAME.match(v):
            raise ValueError(f"joint name {v!r} must match [a-z][a-z0-9_]*")
        return v

    @field_validator("unit")
    @classmethod
    def _unit(cls, v: str) -> str:
        if v not in ("deg", "mm", "rad", "m", "unit"):
            raise ValueError(f"unit {v!r} not one of deg|mm|rad|m|unit")
        return v

    @model_validator(mode="after")
    def _limits(self):
        if self.max <= self.min:
            raise ValueError(f"{self.name}: max ({self.max}) must exceed min ({self.min})")
        if not (self.min <= self.rest <= self.max):
            raise ValueError(f"{self.name}: rest {self.rest} outside [{self.min}, {self.max}]")
        if self.urdf_joint is None:
            self.urdf_joint = self.name
        return self


class MixTerm(BaseModel):
    from_: str = Field(..., alias="from")
    gain: float = 1.0

    model_config = {"populate_by_name": True}


class Spring(BaseModel):
    """Second-order tracker (docs/RETARGET.md §spring): replaces the one-pole
    smoother for this joint. ``hz`` = natural frequency, ``zeta`` = damping
    ratio (1 = critically damped, <1 overshoots)."""
    hz: float = Field(..., gt=0)
    zeta: float = Field(1.0, gt=0, le=2.0)


class Idle(BaseModel):
    """Liveliness layer (docs/RETARGET.md §idle): deterministic band-limited
    sway of peak amplitude ``amp`` (joint units) around ``hz``, added while the
    mapped target is near-still; ``still`` (units/s) is the target speed at
    which it is fully suppressed (default ``10 * amp * hz``)."""
    amp: float = Field(..., ge=0)
    hz: float = Field(..., gt=0)
    still: Optional[float] = Field(None, gt=0)

    def still_speed(self) -> float:
        return self.still if self.still is not None else 10.0 * self.amp * self.hz


class Mapping(BaseModel):
    from_: Optional[str] = Field(None, alias="from")
    gain: float = 1.0
    mix: Optional[List[MixTerm]] = None
    offset: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    deadband: float = 0.0
    smooth_hz: Optional[float] = None
    # v1.1 additive keys (docs/RETARGET.md); all optional, absent = v1 behaviour
    spring: Optional[Spring] = None
    idle: Optional[Idle] = None
    soft_limit: Optional[float] = Field(None, ge=0.0, lt=0.5)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _one_source(self):
        if (self.from_ is None) == (self.mix is None):
            raise ValueError("mapping needs exactly one of `from` or `mix`")
        return self

    def terms(self) -> List[MixTerm]:
        if self.mix is not None:
            return list(self.mix)
        return [MixTerm(**{"from": self.from_, "gain": self.gain})]


class Description(BaseModel):
    urdf: str
    mesh_scale: float = 1.0
    up_axis: str = "z"
    viewer: Dict = Field(default_factory=dict)


class Export(BaseModel):
    formats: List[str] = Field(default_factory=list)
    autonomous_os_csv: Dict = Field(default_factory=dict)
    pollen_move: Dict = Field(default_factory=dict)
    lerobot: Dict = Field(default_factory=dict)


class Runtime(BaseModel):
    kind: str = "none"
    url: Optional[str] = None
    stream_hz: float = 30.0
    extra: Dict = Field(default_factory=dict)


class NativeClips(BaseModel):
    dir: str
    format: str


class Profile(BaseModel):
    schema_: str = Field(..., alias="schema")
    name: str
    display_name: str
    vendor: str = ""
    homepage: str = ""
    license: str = ""
    rate_hz: float = 30.0
    description: Description
    joints: List[Joint]
    retarget: Dict[str, Dict[str, Mapping]]
    export: Export = Field(default_factory=Export)
    runtime: Runtime = Field(default_factory=Runtime)
    native_clips: Optional[NativeClips] = None
    # filled by the loader
    path: str = ""
    prose: str = ""

    model_config = {"populate_by_name": True}

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not _JOINT_NAME.match(v):
            raise ValueError(f"robot name {v!r} must match [a-z][a-z0-9_]*")
        return v

    # ---- lookups ------------------------------------------------------------
    @property
    def dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.path)) if self.path else "."

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)

    def urdf_path(self) -> str:
        return os.path.normpath(os.path.join(self.dir, self.description.urdf))

    def mapping(self, mode: str = "default") -> Dict[str, Mapping]:
        if mode not in self.retarget:
            raise KeyError(f"robot {self.name} has no retarget mode {mode!r}; modes: {list(self.retarget)}")
        return self.retarget[mode]

    # ---- conformance --------------------------------------------------------
    def check(self) -> List[str]:
        """All violations of docs/ROBOT_MD_SPEC.md rules; empty = pass."""
        errs: List[str] = []
        if self.schema_ != SCHEMA:
            errs.append(f"schema must be {SCHEMA}, got {self.schema_!r}")
        names = self.joint_names
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            errs.append(f"duplicate joint names: {dupes}")
        if "default" not in self.retarget:
            errs.append("retarget must define a `default` mode")
        for mode, mp in self.retarget.items():
            for jn, m in mp.items():
                if jn not in names:
                    errs.append(f"retarget.{mode}: {jn!r} is not a declared joint")
                for term in m.terms():
                    if term.from_ not in MAPPABLE:
                        errs.append(f"retarget.{mode}.{jn}: {term.from_!r} is not a canonical channel")
                j = self.joint(jn) if jn in names else None
                if j is not None:
                    lo = j.min if m.min is None else m.min
                    hi = j.max if m.max is None else m.max
                    if lo >= hi:
                        errs.append(f"retarget.{mode}.{jn}: min {lo} >= max {hi}")
                    if m.min is not None and m.min < j.min or m.max is not None and m.max > j.max:
                        errs.append(f"retarget.{mode}.{jn}: mapping bounds exceed joint limits")
                if m.spring is not None and m.spring.hz > self.rate_hz / 4.0:
                    # semi-implicit Euler is stable for w*dt < 2 (~rate/3.1 Hz); keep a margin
                    errs.append(f"retarget.{mode}.{jn}: spring.hz {m.spring.hz} > rate_hz/4 ({self.rate_hz / 4.0}) — unstable at this rate")
                if m.idle is not None and j is not None and m.idle.amp > 0.25 * (j.max - j.min):
                    errs.append(f"retarget.{mode}.{jn}: idle.amp {m.idle.amp} is more than a quarter of the joint range")
        urdf = self.urdf_path()
        if not os.path.exists(urdf):
            errs.append(f"description.urdf not found: {urdf}")
        else:
            text = open(urdf, encoding="utf-8", errors="replace").read()
            present = set(re.findall(r'<joint\s+[^>]*name="([^"]+)"', text))
            limits = urdf_joint_limits(text)
            for j in self.joints:
                if j.urdf_joint not in present:
                    errs.append(f"joint {j.name}: urdf_joint {j.urdf_joint!r} not in {os.path.basename(urdf)}")
                    continue
                lim = limits.get(j.urdf_joint)
                if lim is None:
                    continue
                # profile range → URDF units, then it must lie inside the URDF's own limits (rule 10)
                scale = {"deg": math.pi / 180.0, "mm": 1e-3}.get(j.unit, 1.0)
                a = (j.min + j.urdf_offset) * j.urdf_sign * scale
                b = (j.max + j.urdf_offset) * j.urdf_sign * scale
                lo, hi = min(a, b), max(a, b)
                tol = 1e-3
                if lo < lim[0] - tol or hi > lim[1] + tol:
                    errs.append(f"joint {j.name}: profile range [{j.min}, {j.max}] {j.unit} exceeds the URDF's "
                                f"[{lim[0]:.4f}, {lim[1]:.4f}] on {j.urdf_joint} (rule 10; tighten min/max or fix urdf_sign/offset)")
        if self.native_clips is not None:
            d = os.path.join(self.dir, self.native_clips.dir)
            if not os.path.isdir(d):
                errs.append(f"native_clips.dir not found: {d}")
        return errs

    # ---- export ---------------------------------------------------------------
    def to_web_json(self) -> Dict:
        """Everything the browser viewer needs (paths relative to the robot dir)."""
        return {
            "schema": SCHEMA,
            "name": self.name,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "license": self.license,
            "rate_hz": self.rate_hz,
            "description": self.description.model_dump(),
            "joints": [j.model_dump() for j in self.joints],
            "retarget": {
                mode: {
                    jn: {
                        "terms": [{"from": t.from_, "gain": t.gain} for t in m.terms()],
                        "offset": m.offset,
                        "min": self.joint(jn).min if m.min is None else m.min,
                        "max": self.joint(jn).max if m.max is None else m.max,
                        "deadband": m.deadband,
                        "smooth_hz": m.smooth_hz,
                        "spring": None if m.spring is None else {"hz": m.spring.hz, "zeta": m.spring.zeta},
                        "idle": None if m.idle is None else {"amp": m.idle.amp, "hz": m.idle.hz, "still": m.idle.still_speed()},
                        "soft_limit": m.soft_limit,
                    }
                    for jn, m in mp.items()
                }
                for mode, mp in self.retarget.items()
            },
            "native_clips": self.native_clips.model_dump() if self.native_clips else None,
        }


def urdf_joint_limits(text: str) -> Dict[str, tuple]:
    """``{joint_name: (lower, upper)}`` for every URDF joint carrying a ``<limit>``
    (radians / metres). Continuous joints and joints without limits are absent."""
    out: Dict[str, tuple] = {}
    for m in re.finditer(r'<joint\s+[^>]*name="([^"]+)"[^>]*>(.*?)</joint>', text, re.S):
        lim = re.search(r'<limit\b[^>]*>', m.group(2))
        if not lim:
            continue
        lo = re.search(r'lower="([^"]+)"', lim.group(0))
        hi = re.search(r'upper="([^"]+)"', lim.group(0))
        if lo and hi:
            try:
                out[m.group(1)] = (float(lo.group(1)), float(hi.group(1)))
            except ValueError:
                pass
    return out


def split_front_matter(text: str):
    m = _FRONT.match(text)
    if not m:
        raise ValueError("ROBOT.md must start with a YAML front matter block delimited by ---")
    return m.group(1), m.group(2)


def load_profile(path: str) -> Profile:
    """Load ``robots/<name>/ROBOT.md`` (or a directory containing one)."""
    if os.path.isdir(path):
        path = os.path.join(path, "ROBOT.md")
    text = open(path, encoding="utf-8").read()
    front, prose = split_front_matter(text)
    data = yaml.safe_load(front) or {}
    prof = Profile(**data)
    prof.path = os.path.abspath(path)
    prof.prose = prose.strip()
    return prof


def robots_root(start: Optional[str] = None) -> str:
    """The repo's ``robots/`` directory, found from the package location."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "robots")


def find_robot(name_or_path: str) -> Profile:
    if os.path.exists(name_or_path):
        return load_profile(name_or_path)
    cand = os.path.join(robots_root(), name_or_path)
    if os.path.isdir(cand):
        return load_profile(cand)
    raise FileNotFoundError(f"no robot {name_or_path!r} (looked in {robots_root()})")


def export_web_json(profile: Profile, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profile.to_web_json(), fh, indent=2)
    return out_path
