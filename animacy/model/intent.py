"""Intent conditioning: text -> (arousal, valence, tag), no model at runtime.

Speech-only conditioning cannot know that "No way, that is incredible news!"
needs a burst and "Let me think..." needs a settle. In talk mode the text is
known, so a small, fully documented lexicon + punctuation rule turns it into

* ``tag``      one of ``TAGS`` (greeting, agreement, doubt, excitement, thinking, neutral)
* ``arousal``  0..1, how much energy the line carries
* ``valence``  -1..1, how positive it is

which the generators consume as an amplitude scale (``amplitude_for``:
``0.8 + 0.5 * arousal``, capped) and, for retrieval, as a bonus for human
windows of similar arousal (``RetrievalIndex.query(..., target_arousal=...)``).
``analyse(text, override=...)`` lets ``animacy say --intent excitement`` force a tag.
The rule is deterministic and mirrored verbatim in ``web/models/model.json``
(``intent`` block) so the browser produces the same numbers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

LEXICON_VERSION = "intent.v1"
TAGS = ["greeting", "agreement", "doubt", "excitement", "thinking", "neutral"]

# phrase lists are matched as whole words/phrases on the lower-cased text
LEXICON: Dict[str, List[str]] = {
    "greeting": ["hey", "hi", "hello", "good morning", "good afternoon", "good evening", "welcome", "nice to meet",
                 "good to see", "great to see", "howdy", "greetings", "long time no see"],
    "agreement": ["yes", "yeah", "yep", "exactly", "agreed", "i agree", "sure", "of course", "absolutely", "correct",
                  "indeed", "definitely", "that is what i meant", "that's what i meant", "you're right", "you are right",
                  "makes sense", "precisely"],
    "doubt": ["not sure", "don't think", "do not think", "doubt", "hmm, no", "i disagree", "not really", "that's not right",
              "that is not right", "isn't right", "are you sure", "hardly", "unlikely", "i'm not convinced", "no, i", "no i"],
    "excitement": ["wow", "amazing", "incredible", "awesome", "fantastic", "no way", "can't believe", "cannot believe",
                   "great news", "unbelievable", "brilliant", "love it", "so cool", "yay", "congratulations", "wonderful",
                   "excellent", "oh my", "that's huge", "let's go"],
    "thinking": ["let me think", "hmm", "hm", "i wonder", "what if", "let's see", "let me see", "give me a second",
                 "for a second", "for a moment", "suppose", "consider", "thinking about", "maybe", "perhaps"],
}
# ties: the more expressive tag wins
TAG_PRIORITY = ["excitement", "greeting", "doubt", "agreement", "thinking"]
BASE_AROUSAL = {"greeting": 0.55, "agreement": 0.50, "doubt": 0.35, "excitement": 0.85, "thinking": 0.25, "neutral": 0.40}
BASE_VALENCE = {"greeting": 0.5, "agreement": 0.4, "doubt": -0.4, "excitement": 0.8, "thinking": 0.0, "neutral": 0.0}
POSITIVE = ["good", "great", "love", "happy", "nice", "thanks", "thank you", "glad", "perfect", "beautiful", "fun"]
NEGATIVE = ["bad", "sorry", "terrible", "hate", "wrong", "sad", "awful", "unfortunately", "problem", "afraid"]
EXCLAMATION_AROUSAL = 0.10      # per "!" (max 2 counted)
ELLIPSIS_AROUSAL = -0.10        # a pause ("...") lowers energy
CAPS_AROUSAL = 0.05             # per ALL-CAPS word (max 2 counted)
QUESTION_AROUSAL = 0.05
AMPLITUDE_BASE, AMPLITUDE_GAIN, AMPLITUDE_CAP = 0.8, 0.5, 1.3


@dataclass
class Intent:
    tag: str
    arousal: float
    valence: float
    amplitude: float
    hits: Dict[str, int]
    text: str
    overridden: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def _count(text: str, phrases: List[str]) -> int:
    n = 0
    for p in phrases:
        n += len(re.findall(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z'])", text))
    return n


def amplitude_for(arousal: float) -> float:
    """The amplitude rule: 0.8 at arousal 0, 1.3 at arousal 1 (capped)."""
    return float(min(AMPLITUDE_CAP, AMPLITUDE_BASE + AMPLITUDE_GAIN * max(0.0, min(1.0, float(arousal)))))


def analyse(text: str, override: Optional[str] = None) -> Intent:
    """Text -> Intent. ``override`` forces the tag (its base arousal/valence still get the
    punctuation modifiers)."""
    raw = text or ""
    t = raw.lower().strip()
    hits = {tag: _count(t, LEXICON[tag]) for tag in LEXICON}
    if override:
        if override not in TAGS:
            raise ValueError(f"unknown intent {override!r}; choose from {TAGS}")
        tag, overridden = override, True
    else:
        best = max(hits.values()) if hits else 0
        tag = "neutral"
        if best > 0:
            for cand in TAG_PRIORITY:
                if hits[cand] == best:
                    tag = cand
                    break
        overridden = False
    n_excl = min(2, raw.count("!"))
    n_caps = min(2, sum(1 for w in re.findall(r"[A-Za-z]{2,}", raw) if w.isupper()))
    arousal = BASE_AROUSAL[tag] + EXCLAMATION_AROUSAL * n_excl + CAPS_AROUSAL * n_caps
    if "..." in raw or "…" in raw:
        arousal += ELLIPSIS_AROUSAL
    if "?" in raw:
        arousal += QUESTION_AROUSAL
    arousal = float(max(0.0, min(1.0, arousal)))
    valence = BASE_VALENCE[tag] + 0.1 * min(3, _count(t, POSITIVE)) - 0.1 * min(3, _count(t, NEGATIVE))
    valence = float(max(-1.0, min(1.0, valence)))
    return Intent(tag=tag, arousal=arousal, valence=valence, amplitude=amplitude_for(arousal), hits=hits, text=raw,
                  overridden=overridden)


GRADER_LINES = {
    "greeting": "Hey! Good to see you again.",
    "agreement": "Yes, exactly, that is what I meant.",
    "doubt": "Hmm, no, I really don't think that's right.",
    "excitement": "No way, that is incredible news!",
    "thinking": "Let me think about that for a second... okay.",
}


def grader_lines() -> Dict[str, str]:
    """The five grader movements' utterances (from ``animacy.grade.movements`` when importable)."""
    try:
        from ..grade.movements import MOVEMENTS  # type: ignore

        return {m.key: m.text for m in MOVEMENTS}
    except Exception:  # noqa: BLE001 - the grader package is optional here
        return dict(GRADER_LINES)


def describe(arousal_weight: float, thinking_weight: float) -> Dict:
    """Everything the browser needs to reproduce the rule, plus what it says about the grader lines."""
    return {
        "lexicon_version": LEXICON_VERSION,
        "tags": TAGS,
        "lexicon": LEXICON,
        "tag_priority_on_ties": TAG_PRIORITY,
        "base_arousal": BASE_AROUSAL,
        "base_valence": BASE_VALENCE,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "rules": {
            "match": "whole-word / whole-phrase matches on the lower-cased text; the tag with most hits wins, ties by tag_priority_on_ties; no hits = neutral",
            "arousal": f"base_arousal[tag] + {EXCLAMATION_AROUSAL} per '!' (max 2) + {CAPS_AROUSAL} per ALL-CAPS word (max 2) "
                       f"{ELLIPSIS_AROUSAL:+} if '...' present {QUESTION_AROUSAL:+} if '?' present, clamped to [0, 1]",
            "valence": "base_valence[tag] + 0.1 per positive word (max 3) - 0.1 per negative word (max 3), clamped to [-1, 1]",
            "amplitude": f"min({AMPLITUDE_CAP}, {AMPLITUDE_BASE} + {AMPLITUDE_GAIN} * arousal), applied to the decoded canonical motion of every source",
            "retrieval_bonus": f"{arousal_weight} * (1 - |window_arousal - target_arousal|) added to the cosine score of every index window",
            "thinking_bonus": f"{thinking_weight} * max(0, still_then_move) when tag == thinking (windows whose second half moves more than their first)",
            "override": "animacy say --intent <tag> forces the tag; punctuation modifiers still apply",
        },
        "grader_lines": {k: {"text": v, **{kk: vv for kk, vv in analyse(v).to_dict().items() if kk in ("tag", "arousal", "valence", "amplitude")}}
                         for k, v in grader_lines().items()},
    }
