"""The five movements and how every clip for them is built.

Candidate clips come from ``animacy.serve``'s motion sources exactly as the
talk loop uses them (TTS waveform -> source -> ``retarget_clip`` through the
robot's ``ROBOT.md``), one per (robot, movement, source, seed). Calibration
clips are the vendors' own hand-authored clips for the same intents, loaded
from ``robots/<robot>/clips/native`` and rendered through the same pipeline.

Blindness note: a vendor clip has no speech, so its card says "The robot
expresses: <intent>" while a candidate's says "The robot says: <sentence>".
Neither card says how the motion was made.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..export import read_autonomous_os_csv
from ..profile import Profile, find_robot
from ..retarget import retarget_clip
from ..serve import SOURCES
from .render import ROOT

VENDOR = "vendor"
TUNING_SET = "tuning"      # the five lines in this file: known to every agent, so a model can be tuned to them
HELDOUT_SET = "heldout"    # five sealed lines authored by the grader: the gate from run 2 on
HELDOUT_PATH = os.path.join(ROOT, "data", "grading", "heldout_lines.json")
DEFAULT_CHECKPOINT = os.path.join(ROOT, "checkpoints", "v1")
VENDOR_MAX_SECONDS = 6.0     # long vendor clips (thinking_deep is 15 s) are trimmed so reels stay short


@dataclass(frozen=True)
class Movement:
    key: str
    text: str                      # the utterance a candidate clip is generated from
    expression: str                # what the silent calibration clip expresses (card text)
    vendor: Dict[str, str]         # robot -> native clip name
    intent: Optional[str] = None   # tag handed to a motion source that accepts ``intent`` (defaults to key)
    line_set: str = TUNING_SET     # tuning (these lines) | heldout (sealed lines with the same intents)

    @property
    def intent_tag(self) -> str:
        return self.intent or self.key

    @property
    def label(self) -> str:
        """Movement segment of a clip id: ``greeting`` for the tuning set, ``greeting@heldout`` otherwise."""
        return self.key if self.line_set == TUNING_SET else f"{self.key}@{self.line_set}"


MOVEMENTS: List[Movement] = [
    Movement("greeting", "Hey! Good to see you again.", "a greeting",
             {"lamp": "greeting", "reachy_mini": "welcoming1"}),
    Movement("agreement", "Yes, exactly, that is what I meant.", "agreement, a nod",
             {"lamp": "nod", "reachy_mini": "yes1"}),
    Movement("doubt", "Hmm, no, I really don't think that's right.", "disagreement, a head shake",
             {"lamp": "headshake", "reachy_mini": "no1"}),
    Movement("excitement", "No way, that is incredible news!", "excitement",
             {"lamp": "excited", "reachy_mini": "amazed1"}),
    # lamp: the vendor's thinking_deep.csv is a constant pose (every joint range 0.0 over 15 s), so the
    # nearest expressive vendor clip for "thinking it over" is confused (8 s, head tilts and searches).
    Movement("thinking", "Let me think about that for a second... okay.", "thinking it over",
             {"lamp": "confused", "reachy_mini": "thoughtful1"}),
]
MOVEMENT_KEYS = [m.key for m in MOVEMENTS]


def movement(key: str) -> Movement:
    for m in MOVEMENTS:
        if m.key == key:
            return m
    raise KeyError(key)


def _words(text: str) -> List[str]:
    return [w.strip("'") for w in re.findall(r"[a-z']+", text.lower())]


def shared_phrases(a: str, b: str, n: int = 3) -> List[str]:
    """Word n-grams (default 3) that ``a`` and ``b`` share: the 'no phrase reused' check for held-out lines."""
    wa, wb = _words(a), _words(b)
    ga = {" ".join(wa[i:i + n]) for i in range(len(wa) - n + 1)}
    gb = {" ".join(wb[i:i + n]) for i in range(len(wb) - n + 1)}
    return sorted(ga & gb)


def load_heldout_movements(path: str = HELDOUT_PATH, base: Sequence[Movement] = MOVEMENTS) -> List[Movement]:
    """The sealed held-out lines (``{"lines": {intent: text}}``) as Movements with ``line_set = heldout``, one per
    base movement (same intent, expression and vendor clip). Missing file -> empty list. The texts are never
    logged or printed by this package."""
    if not os.path.isfile(path):
        return []
    obj = json.load(open(path, encoding="utf-8"))
    lines = obj.get("lines") or {}
    out: List[Movement] = []
    for mv in base:
        text = lines.get(mv.key)
        if not text or not str(text).strip():
            raise ValueError(f"held-out file {path} has no line for intent {mv.key!r}")
        for other in base:
            if shared_phrases(text, other.text):
                raise ValueError(f"held-out line for {mv.key!r} shares a 3-word phrase with the tuning line for {other.key!r}")
        out.append(Movement(mv.key, str(text).strip(), mv.expression, mv.vendor, mv.intent, line_set=HELDOUT_SET))
    return out


@dataclass
class ClipSpec:
    """One clip to be rendered and judged. ``table`` is in robot units on the robot's grid."""

    id: str
    robot: str
    movement: str
    source: str                          # model | retrieval | envelope | vendor
    seed: Optional[int]
    card_line: str
    table: pd.DataFrame
    audio: Optional[np.ndarray] = None
    sr: int = 16000
    vendor_clip: Optional[str] = None
    meta: Dict = field(default_factory=dict)
    line_set: str = TUNING_SET

    @property
    def duration(self) -> float:
        return float(self.table["t"].iloc[-1]) if len(self.table) else 0.0

    def public(self) -> Dict:
        """Everything but the table/audio (what the sealed manifest records)."""
        return {"id": self.id, "robot": self.robot, "movement": self.movement, "source": self.source,
                "seed": self.seed, "vendor_clip": self.vendor_clip, "card_line": self.card_line,
                "line_set": self.line_set, "duration": round(self.duration, 3), "meta": self.meta}


# ---------------------------------------------------------------- vendor clips
def load_vendor_table(profile: Profile, clip_name: str, max_seconds: Optional[float] = VENDOR_MAX_SECONDS) -> pd.DataFrame:
    if profile.native_clips is None:
        raise ValueError(f"{profile.name} declares no native_clips in ROBOT.md")
    d = os.path.join(profile.dir, profile.native_clips.dir)
    fmt = profile.native_clips.format
    if fmt == "autonomous_os_csv":
        table = read_autonomous_os_csv(os.path.join(d, clip_name + ".csv"))
    elif fmt == "json":
        obj = json.load(open(os.path.join(d, clip_name + ".json"), encoding="utf-8"))
        t = np.asarray(obj["t"], dtype=np.float64)
        table = pd.DataFrame({"t": t - t[0], **{j: np.asarray(v, dtype=np.float64) for j, v in obj["data"].items()}})
    else:
        raise ValueError(f"unsupported native clip format {fmt!r}")
    if max_seconds is not None:
        table = most_active_window(table, max_seconds)
    return table


def most_active_window(table: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """The ``seconds``-long window with the most joint motion (sum of |delta| over joints), ``t`` re-zeroed.
    A clip shorter than ``seconds`` is returned whole. Long vendor clips often start or end with a hold;
    the first N seconds of ``excited`` is mostly a hold, its middle is the gesture."""
    t = table["t"].to_numpy(dtype=np.float64)
    if not len(t) or t[-1] - t[0] <= seconds + 1e-9:
        return table.reset_index(drop=True)
    joints = [c for c in table.columns if c != "t"]
    v = np.abs(np.diff(table[joints].to_numpy(dtype=np.float64), axis=0)).sum(axis=1)   # motion between rows
    cum = np.concatenate([[0.0], np.cumsum(v)])                                      # cum[i] = motion up to row i
    best_start, best = 0, -1.0
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + seconds, side="right")) - 1
        if j <= i:
            continue
        score = cum[j] - cum[i]
        if score > best:
            best, best_start = score, i
        if t[i] + seconds > t[-1]:
            break
    j = int(np.searchsorted(t, t[best_start] + seconds, side="right")) - 1
    out = table.iloc[best_start:j + 1].reset_index(drop=True).copy()
    out["t"] = out["t"] - out["t"].iloc[0]
    return out


def vendor_clip(profile: Profile, mv: Movement, max_seconds: Optional[float] = VENDOR_MAX_SECONDS) -> ClipSpec:
    name = mv.vendor[profile.name]
    return ClipSpec(id=f"{profile.name}/{mv.key}/{VENDOR}/{name}", robot=profile.name, movement=mv.key, source=VENDOR,
                    seed=None, card_line=f"The robot expresses: {mv.expression}",
                    table=load_vendor_table(profile, name, max_seconds), audio=None, vendor_clip=name)


# ---------------------------------------------------------------- candidates
def _wav_cache_path(cache_dir: str, text: str, engine: str) -> str:
    h = hashlib.sha1(f"{engine}|{text}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"tts_{h}.wav")


def synth_cached(text: str, cache_dir: str, engine: str = "auto") -> Tuple[np.ndarray, int]:
    """``animacy.tts.synth`` with a wav cache (one TTS call per utterance per run)."""
    import soundfile as sf

    from ..tts import synth

    os.makedirs(cache_dir, exist_ok=True)
    path = _wav_cache_path(cache_dir, text, engine)
    if os.path.exists(path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1), sr
    wav, sr = synth(text, engine=engine)
    sf.write(path, wav, sr)
    return wav, sr


# ---------------------------------------------------------------- source variants (A/B knobs)
@dataclass(frozen=True)
class Variant:
    """An extra graded column: ``base`` source called with ``kwargs`` (e.g. retrieval with proto_weight=0).
    A knob is applied only when it is an explicit parameter of the source function; a ``**kw`` catch-all does
    not count, because a swallowed knob would make the A/B a silent no-op."""
    name: str
    base: str
    kwargs: Dict[str, float]


def parse_variant(spec: str) -> Variant:
    """``name=base:key=value[,key=value]`` -> Variant (values are floats, or ints when integral)."""
    try:
        name, rest = spec.split("=", 1)
        base, kv = rest.split(":", 1)
    except ValueError as e:
        raise ValueError(f"variant {spec!r}: expected name=base:key=value[,key=value]") from e
    kwargs: Dict[str, float] = {}
    for item in kv.split(","):
        k, v = item.split("=", 1)
        x = float(v)
        kwargs[k.strip()] = int(x) if x.is_integer() else x
    name, base = name.strip(), base.strip()
    if base not in SOURCES:
        raise ValueError(f"variant {name!r}: base source {base!r} not in {list(SOURCES)}")
    if name in SOURCES:
        raise ValueError(f"variant name {name!r} collides with a source name")
    return Variant(name, base, kwargs)


def explicit_params(fn) -> set:
    try:
        return {n for n, p in inspect.signature(fn).parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD}
    except (TypeError, ValueError):
        return set()


def accepts_intent(fn) -> bool:
    """Does a motion source take an ``intent`` keyword (the talk loop's intent conditioning)?"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "intent" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def intent_argument(text: str, tag: Optional[str]):
    """What ``serve.say(text, intent=tag)`` hands the source: ``analyse(text, override=tag)`` (the tag's base
    arousal plus the line's punctuation) when the intent module exists, else the bare tag."""
    try:
        from ..model.intent import analyse
    except ImportError:
        return tag
    return analyse(text, override=tag) if tag else text


def candidate_table(profile: Profile, source: str, wav: np.ndarray, sr: int, seed: int,
                    checkpoint: str = DEFAULT_CHECKPOINT, intent: Optional[str] = None, text: Optional[str] = None,
                    sources: Optional[Dict] = None, variant: Optional[Variant] = None) -> Tuple[pd.DataFrame, Dict]:
    """Exactly ``serve.say``'s dispatch, minus the sink: source -> HumanClip -> retarget_clip.
    The intent (``intent`` tag + the utterance ``text``) is handed to sources whose signature accepts it, in the
    same form ``animacy say --intent <tag>`` uses; the envelope heuristic and older signatures are untouched."""
    fn = (sources or SOURCES)[source]
    kwargs: Dict = {"seed": seed}
    if source != "envelope":
        kwargs["checkpoint"] = checkpoint
    variant_applied: Dict[str, float] = {}
    if variant is not None:
        allowed = explicit_params(fn)
        variant_applied = {k: v for k, v in variant.kwargs.items() if k in allowed}
        kwargs.update(variant_applied)
    passed_intent = source != "envelope" and bool(intent) and accepts_intent(fn)
    if passed_intent:
        kwargs["intent"] = intent_argument(text or "", intent)
    clip = fn(wav, sr, **kwargs)
    probs = clip.validate()
    if probs:
        raise RuntimeError(f"motion source {source!r} produced an invalid clip: {probs}")
    table = retarget_clip(clip, profile)
    meta = {k: v for k, v in clip.meta.items() if isinstance(v, (str, int, float, bool))}
    meta["intent_passed"] = intent if passed_intent else None
    if variant is not None:
        meta["variant"] = {"name": variant.name, "base": variant.base, "requested": dict(variant.kwargs),
                           "applied": variant_applied, "no_op": not variant_applied}
    if isinstance(clip.meta.get("intent"), dict):
        meta["intent"] = {k: v for k, v in clip.meta["intent"].items() if isinstance(v, (str, int, float, bool))}
    return table, meta


def candidate_clip(profile: Profile, mv: Movement, source: str, seed: int, wav: np.ndarray, sr: int,
                   checkpoint: str = DEFAULT_CHECKPOINT, variant: Optional[Variant] = None) -> ClipSpec:
    """``variant`` makes the clip's ``source`` the variant's name (an extra column) while the motion comes from the
    variant's base source with its knobs."""
    base = variant.base if variant is not None else source
    table, meta = candidate_table(profile, base, wav, sr, seed, checkpoint, intent=mv.intent_tag, text=mv.text, variant=variant)
    return ClipSpec(id=f"{profile.name}/{mv.label}/{source}/s{seed}", robot=profile.name, movement=mv.key, source=source,
                    seed=seed, card_line=f'The robot says: "{mv.text}"', table=table, audio=wav, sr=sr, meta=meta,
                    line_set=mv.line_set)


DETERMINISTIC_SOURCES = ("retrieval", "envelope")   # retrieval ignores the seed; envelope's seed only shifts slow drift phases


def build_clips(robots: Sequence[str], sources: Sequence[str], seeds: int, run_dir: str,
                checkpoint: str = DEFAULT_CHECKPOINT, movements: Sequence[Movement] = MOVEMENTS,
                with_vendor: bool = True, tts_engine: str = "auto", seeds_by_source: Optional[Dict[str, int]] = None,
                heldout: Optional[Sequence[Movement]] = None, variants: Optional[Sequence[Variant]] = None,
                log=print) -> List[ClipSpec]:
    """Every clip for the run: candidates for each (robot, movement, source, seed) + one vendor clip per
    (robot, movement). ``seeds_by_source`` overrides the seed count per source (deterministic sources need one).
    Joint tables and utterance audio are saved under ``<run_dir>/clips`` for the record."""
    seeds_by_source = dict(seeds_by_source or {})
    variants = {v.name: v for v in (variants or [])}
    bad = [s for s in sources if s not in SOURCES and s not in variants]
    if bad:
        raise ValueError(f"unknown sources {bad}; choose from {list(SOURCES)}")
    clips: List[ClipSpec] = []
    tts_dir = os.path.join(run_dir, "tts")
    clip_dir = os.path.join(run_dir, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    all_movements = list(movements) + list(heldout or [])
    wavs = {(mv.line_set, mv.key): synth_cached(mv.text, tts_dir, tts_engine) for mv in all_movements}
    for rname in robots:
        profile = find_robot(rname)
        for mv in all_movements:
            wav, sr = wavs[(mv.line_set, mv.key)]
            for source in sources:
                v = variants.get(source)
                n_seeds = seeds_by_source.get(source, seeds_by_source.get(v.base, seeds) if v else seeds)
                for seed in range(n_seeds):
                    c = candidate_clip(profile, mv, source, seed, wav, sr, checkpoint, variant=v)
                    clips.append(c)
                    log(f"[clips] {c.id}: {len(c.table)} frames, {c.duration:.2f}s")
            if with_vendor and mv.line_set == TUNING_SET:      # one calibration clip per intent, shared by both sets
                c = vendor_clip(profile, mv)
                clips.append(c)
                log(f"[clips] {c.id}: {len(c.table)} frames, {c.duration:.2f}s (vendor)")
    for c in clips:
        p = os.path.join(clip_dir, c.id.replace("/", "__") + ".json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({**c.public(), "t": c.table["t"].round(4).tolist(),
                       "data": {k: c.table[k].round(3).tolist() for k in c.table.columns if k != "t"}}, fh)
    return clips
