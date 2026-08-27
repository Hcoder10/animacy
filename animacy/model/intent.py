"""Intent conditioning: text -> (arousal, valence, tag), no model at runtime.

Speech-only conditioning cannot know that an exclamation of amazement needs a
burst and a pondering line needs a settle. In talk mode the text is known, so a
small, fully documented lexicon of GENERIC cue words + punctuation rules turns
it into

* ``tag``      one of ``TAGS`` (greeting, agreement, doubt, excitement, thinking, neutral)
* ``arousal``  0..1, how much energy the line carries
* ``valence``  -1..1, how positive it is

which the generators consume as an amplitude scale (``amplitude_for``:
``0.8 + 0.5 * arousal``, capped) and, for retrieval, as a bonus for human
windows of similar arousal (``RetrievalIndex.query(..., target_arousal=...)``).
``analyse(text, override=...)`` lets ``animacy say --intent excitement`` force a tag.

Integrity: the lexicon contains no utterance of the blind grader; it is cue
words any greeting / agreement / doubt / excitement / thinking line shares.
``EXAMPLE_LINES`` are ten lines written for this module to show what the rule
does. The rule is deterministic and mirrored verbatim in
``web/models/model.json`` (``intent`` block) so the browser produces the same numbers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

LEXICON_VERSION = "intent.v2"
TAGS = ["greeting", "agreement", "doubt", "excitement", "thinking", "neutral"]

# generic cue words / phrases per tag, matched as whole words or phrases on the lower-cased text
LEXICON: Dict[str, List[str]] = {
    "greeting": ["hi", "hey", "hello", "howdy", "welcome", "good morning", "good afternoon", "good evening",
                 "good to see", "nice to meet", "nice to see", "great to see", "greetings", "long time"],
    "agreement": ["yes", "yeah", "yep", "exactly", "right", "agree", "agreed", "of course", "sure", "absolutely",
                  "correct", "indeed", "definitely", "makes sense", "precisely", "true", "fair enough"],
    "doubt": ["no", "nope", "nah", "not sure", "not true", "don't think", "do not think", "doubt", "disagree", "not really",
              "really?", "are you sure", "hardly", "unlikely", "not convinced", "wrong", "i don't know", "hmm"],
    "excitement": ["wow", "no way", "incredible", "amazing", "awesome", "fantastic", "unbelievable", "can't believe",
                   "cannot believe", "brilliant", "wonderful", "excellent", "so cool", "love it", "yay", "congratulations",
                   "oh my", "let's go", "!!"],
    "thinking": ["let me think", "let me see", "let's see", "wait", "consider", "i wonder", "what if", "suppose",
                 "hold on", "give me a moment", "thinking", "maybe", "perhaps", "hmm..."],
}
# ties: the more expressive tag wins
TAG_PRIORITY = ["excitement", "greeting", "doubt", "agreement", "thinking"]
BASE_AROUSAL = {"greeting": 0.55, "agreement": 0.50, "doubt": 0.35, "excitement": 0.85, "thinking": 0.25, "neutral": 0.40}
BASE_VALENCE = {"greeting": 0.5, "agreement": 0.4, "doubt": -0.4, "excitement": 0.8, "thinking": 0.0, "neutral": 0.0}
POSITIVE = ["good", "great", "love", "happy", "nice", "thanks", "thank you", "glad", "perfect", "beautiful", "fun", "news"]
NEGATIVE = ["bad", "sorry", "terrible", "hate", "wrong", "sad", "awful", "unfortunately", "problem", "afraid"]
EXCLAMATION_AROUSAL = 0.10      # per "!" (max 2 counted)
ELLIPSIS_AROUSAL = -0.10        # a pause ("...") lowers energy and counts as a thinking cue
CAPS_AROUSAL = 0.05             # per ALL-CAPS word (max 2 counted)
QUESTION_AROUSAL = 0.05
AMPLITUDE_BASE, AMPLITUDE_GAIN, AMPLITUDE_CAP = 0.8, 0.5, 1.3

# ten lines written for this module (two per intent, none shared with any grader)
EXAMPLE_LINES: Dict[str, List[str]] = {
    "greeting": ["Hello there, welcome back!", "Hi, nice to meet you."],
    "agreement": ["Yeah, I agree, that makes sense.", "Of course, you're right about that."],
    "doubt": ["I'm not sure that's true, honestly.", "Hmm, really? I doubt it."],
    "excitement": ["Wow, this is amazing, I can't believe it!", "That's incredible, congratulations!"],
    "thinking": ["Hmm... let me see, what if we tried the other one.", "Wait, I need to consider that for a moment."],
}


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


NEGATION = r"(?<!not )(?<!n't )(?<!never )"      # "not sure" / "isn't right" do not count as agreement


def _count(text: str, phrases: List[str], negation_aware: bool = False) -> int:
    n = 0
    for p in phrases:
        if p == "!!":
            n += 1 if "!!" in text else 0
        elif p == "hmm...":
            n += len(re.findall(r"hmm+\s*(?:\.\.\.|…)", text))
        elif p == "really?":
            n += len(re.findall(r"really\s*\?", text))
        else:
            n += len(re.findall((NEGATION if negation_aware else "") + r"(?<![a-z'])" + re.escape(p) + r"(?![a-z'])", text))
    return n


def amplitude_for(arousal: float) -> float:
    """The amplitude rule: 0.8 at arousal 0, 1.3 at arousal 1 (capped)."""
    return float(min(AMPLITUDE_CAP, AMPLITUDE_BASE + AMPLITUDE_GAIN * max(0.0, min(1.0, float(arousal)))))


def analyse(text: str, override: Optional[str] = None) -> Intent:
    """Text -> Intent. ``override`` forces the tag (its base arousal/valence still get the
    punctuation modifiers)."""
    raw = text or ""
    t = raw.lower().strip()
    hits = {tag: _count(t, LEXICON[tag], negation_aware=(tag == "agreement")) for tag in LEXICON}
    if "..." in raw or "…" in raw:
        hits["thinking"] += 1                    # a written pause is a thinking cue
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


def example_table() -> Dict[str, List[Dict]]:
    """What the rule says about ``EXAMPLE_LINES`` (its own lines, for documentation)."""
    out: Dict[str, List[Dict]] = {}
    for tag, lines in EXAMPLE_LINES.items():
        out[tag] = [{"text": s, **{k: v for k, v in analyse(s).to_dict().items() if k in ("tag", "arousal", "valence", "amplitude")}}
                    for s in lines]
    return out


def describe(arousal_weight: float, thinking_weight: float) -> Dict:
    """Everything the browser needs to reproduce the rule."""
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
            "match": "whole-word / whole-phrase matches on the lower-cased text ('!!' = the string, 'really?' and 'hmm...' allow "
                     "spaces before the punctuation); agreement cues preceded by 'not ', \"n't \" or 'never ' do not count; "
                     "a written pause '...' counts as one thinking hit; the tag with most hits wins, ties by "
                     "tag_priority_on_ties; no hits = neutral",
            "arousal": f"base_arousal[tag] + {EXCLAMATION_AROUSAL} per '!' (max 2) + {CAPS_AROUSAL} per ALL-CAPS word (max 2) "
                       f"{ELLIPSIS_AROUSAL:+} if '...' present {QUESTION_AROUSAL:+} if '?' present, clamped to [0, 1]",
            "valence": "base_valence[tag] + 0.1 per positive word (max 3) - 0.1 per negative word (max 3), clamped to [-1, 1]",
            "amplitude": f"min({AMPLITUDE_CAP}, {AMPLITUDE_BASE} + {AMPLITUDE_GAIN} * arousal), applied to the decoded canonical motion of every source",
            "retrieval_bonus": f"{arousal_weight} * (1 - |window_arousal - target_arousal|) added to the cosine score of every index window",
            "thinking_bonus": f"{thinking_weight} * max(0, still_then_move) when tag == thinking (windows whose second half moves more than their first)",
            "override": "animacy say --intent <tag> forces the tag; punctuation modifiers still apply",
        },
        "examples": example_table(),
    }
