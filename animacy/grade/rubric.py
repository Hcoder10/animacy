"""The judge's prompt. It must carry ZERO project context.

Nothing in here may say (or hint) how any clip was produced: no project name,
no method words, no origin words. ``FORBIDDEN`` is the enforced list and
``tests/test_grade.py`` checks every prompt this module can build against it.
The judge is asked to describe each clip before scoring it, so the
description proves it watched, and to score each clip on its own.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

# lower-case substrings that must never appear in a prompt (checked case-insensitively)
FORBIDDEN: List[str] = [
    "animacy", "model", "retrieval", "generated", "generate", "vendor", "learned", "learn", "neural",
    "network", "training", "trained", "envelope", "heuristic", "baseline", "dataset", "canonical",
    "retarget", "seed", "checkpoint", "ground truth", "hand-authored", "hand authored", "synthetic",
    "algorithm", "source", "candidate", "calibration", "reference clip", "manifest", "blind", "onnx",
    "transformer", "codebook", "vq", "tts", "text-to-speech", "text to speech", "kokoro", "sapi",
]

DIMENSIONS: List[str] = ["lifelike", "intent", "timing", "physical", "appeal"]
SCORE_KEYS: List[str] = DIMENSIONS + ["overall"]

ROBOT_DESCRIPTIONS: Dict[str, str] = {
    "lamp": "a small desk-lamp robot: a five-joint arm on a square base with a lamp-shade head that looks where it points",
    "reachy_mini": "a small desk robot: a rounded head on a short cylindrical body, with two thin antenna ears on top",
}


def robot_description(robot: str) -> str:
    return ROBOT_DESCRIPTIONS.get(robot, "a small expressive desk robot")


def build_prompt(reel_path: str, clip_numbers: Sequence[int], robot: str, has_speech_strip: bool = True) -> str:
    nums = list(clip_numbers)
    first, last = nums[0], nums[-1]
    listed = ", ".join(str(n) for n in nums)
    strip = (
        "A thin strip along the bottom edge of every frame shows the loudness of the robot's voice over "
        "time, with a bright marker at the current moment; a flat strip means the robot is silent in that clip. "
        "Use it to judge whether the movement phrases line up with the voice, since you may not be able to hear the audio track. "
        if has_speech_strip else ""
    )
    return f"""You are an expert character animator judging short video clips of a small expressive desk robot.
The robot is {robot_description(robot)}.

Watch the ENTIRE video file at {reel_path}. It contains {len(nums)} clips, numbered {first} to {last} ({listed}).
Each clip starts with a title card that shows its number ("Clip N") and one line saying either the sentence the
robot is speaking during the clip or the expression it is performing. Clips are separated by short black gaps.
{strip}
If you can only sample frames from the video, sample as densely as you can and look at every clip; note in
"notes" how many frames you saw.

Judge every clip independently, on its own merits, as a piece of character animation for this body. Do not
compare clips with each other, do not rank them, and do not assume anything about how any clip was made.

First DESCRIBE what the robot actually did in the clip (which parts moved, roughly how far and how fast, and when
relative to the speech or expression). Then score it from 1 to 10 on each dimension:

- lifelike: does it move like a living thing? Anticipation before a move, overshoot and settle after it,
  small continuous secondary motion, variation between repeats. Mechanical, jittery, frozen or metronomic
  motion scores low.
- intent: does the movement read as the stated sentence or expression? Would a viewer name the expression
  correctly without the card?
- timing: do the phrases of motion align with the rhythm of the speech (or the natural rhythm of the expression)?
  Motion that ignores the voice, or is uniform throughout, scores low.
- physical: physically plausible for this body: no impossible speeds or teleports, no jitter or trembling,
  no shaking on a hold.
- appeal: would you enjoy watching this character? Clarity, charm, staging.
- overall: your single judgement of the clip as a whole, 1 to 10.

Anchor the scale: 10 = as good as a top studio's hand-keyed character animation for this body; 8 = clearly alive
and convincing with only minor flaws; 6 = readable but somewhat mechanical or generic; 4 = barely related to the
stated intent, or distractingly unnatural; 1-2 = frozen, jittery, broken or off-body.

Return ONLY a JSON object, no markdown, in exactly this shape, with exactly one entry per clip number {first}..{last}:
{{"clips": [
  {{"clip": {first}, "description": "one or two sentences of what the robot did",
   "scores": {{"lifelike": 0, "intent": 0, "timing": 0, "physical": 0, "appeal": 0, "overall": 0}},
   "reason": "one sentence justifying the overall score"}}
], "notes": "how many frames you saw and anything that limited your judgement"}}
Integers from 1 to 10 only. Do not modify any files."""


def forbidden_hits(text: str, words: Sequence[str] = FORBIDDEN) -> List[str]:
    """Which forbidden words occur in ``text`` (case-insensitive substring match)."""
    low = text.lower()
    return [w for w in words if w in low]


def validate_response(obj: Dict, clip_numbers: Sequence[int]) -> List[str]:
    """Problems with a judge response for ``clip_numbers``; empty = usable."""
    errs: List[str] = []
    clips = obj.get("clips")
    if not isinstance(clips, list):
        return ["no 'clips' list"]
    seen: Dict[int, Dict] = {}
    for entry in clips:
        if not isinstance(entry, dict):
            errs.append("non-object entry")
            continue
        try:
            n = int(entry.get("clip"))
        except (TypeError, ValueError):
            errs.append(f"entry without a clip number: {str(entry)[:80]}")
            continue
        if n in seen:
            errs.append(f"clip {n} scored twice")
        seen[n] = entry
        scores = _scores_of(entry)
        for k in SCORE_KEYS:
            v = scores.get(k)
            if not isinstance(v, (int, float)) or not 1 <= float(v) <= 10:
                errs.append(f"clip {n}: score {k!r} missing or out of range: {v!r}")
        if not str(entry.get("description", "")).strip():
            errs.append(f"clip {n}: no description")
    missing = [n for n in clip_numbers if n not in seen]
    if missing:
        errs.append(f"missing clips: {missing}")
    extra = [n for n in seen if n not in set(clip_numbers)]
    if extra:
        errs.append(f"unexpected clips: {extra}")
    return errs


def _scores_of(entry: Dict) -> Dict:
    """The entry's scores. A judge sometimes puts ``overall`` next to ``scores`` instead of inside it; that is
    accepted (parsing leniency only: every score is still required to be a number in 1..10)."""
    s = dict(entry.get("scores") or {})
    if "overall" not in s and isinstance(entry.get("overall"), (int, float)):
        s["overall"] = entry["overall"]
    return s


def normalise_scores(entry: Dict) -> Dict[str, float]:
    s = _scores_of(entry)
    return {k: float(s[k]) for k in SCORE_KEYS if k in s and isinstance(s[k], (int, float))}


_WS = re.compile(r"\s+")


def one_line(text: str, limit: int = 240) -> str:
    t = _WS.sub(" ", str(text)).strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."
