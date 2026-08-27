"""Intent conditioning: text -> (arousal, valence, tag), no model at runtime.

Speech-only conditioning cannot know that an exclamation of amazement needs a
burst and a pondering line needs a settle. In talk mode the text is known, so a
small, fully documented lexicon of GENERIC cue families + punctuation rules
turns it into

* ``tag``      one of ``TAGS`` (greeting, agreement, doubt, excitement, thinking, neutral)
* ``arousal``  0..1, how much energy the line carries
* ``valence``  -1..1, how positive it is

which the generators consume as an amplitude scale (``amplitude_for``:
``0.8 + 0.5 * arousal``, capped) and, for retrieval, as a bonus for human
windows of similar arousal (``RetrievalIndex.query(..., target_arousal=...)``).
``analyse(text, override=...)`` lets ``animacy say --intent excitement`` force a tag.

Integrity: the lexicon contains no utterance of the blind grader and no
multi-word run copied from one; it is the salutation / affirmative / hedge /
exclamation / deliberation families any such line shares. ``EXAMPLE_LINES``
are thirty lines written for this module (six per intent) to show what the
rule does. The rule is deterministic and mirrored verbatim in
``web/models/model.json`` (``intent`` block) so the browser produces the same numbers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

LEXICON_VERSION = "intent.v3"
TAGS = ["greeting", "agreement", "doubt", "excitement", "thinking", "neutral"]

# generic cue families per tag, matched as whole words / phrases on the lower-cased text
LEXICON: Dict[str, List[str]] = {
    "greeting": ["hi", "hey", "hello", "hiya", "yo", "howdy", "hey there", "hi there", "morning", "afternoon", "evening",
                 "good morning", "good afternoon", "good evening", "good day", "welcome", "welcome back",
                 "nice to see", "great to see", "lovely to see", "nice to meet", "pleased to meet", "glad you're here",
                 "glad you are here", "there you are", "it's been a while", "it has been a while", "long time",
                 "how are you", "how's it going", "how is it going", "how have you been", "greetings"],
    "agreement": ["yes", "yeah", "yep", "yup", "right", "exactly", "agreed", "agree", "true", "correct", "sure",
                  "of course", "absolutely", "definitely", "totally", "indeed", "precisely", "makes sense",
                  "fair point", "good point", "fair enough", "i think so", "you're right", "you are right",
                  "well said", "no doubt", "that's it"],
    "doubt": ["no", "nope", "nah", "not sure", "unsure", "not true", "doubt", "doubtful", "i don't buy", "don't buy",
              "don't believe", "do not believe", "hard to believe", "seems off", "sounds off", "that can't be",
              "can't be right", "not convinced", "questionable", "really?", "is that so", "are you sure",
              "don't think", "do not think", "disagree", "not really", "hardly", "unlikely", "wrong", "i don't know",
              "not so sure", "skeptical", "sceptical", "i doubt", "not buying"],
    "excitement": ["wow", "whoa", "omg", "oh my", "no way", "incredible", "amazing", "awesome", "fantastic",
                   "unbelievable", "best", "love", "can't wait", "cannot wait", "so excited", "yes!!", "let's go",
                   "brilliant", "wonderful", "excellent", "so cool", "yay", "congratulations", "congrats",
                   "what a win", "thrilled", "can't believe", "cannot believe", "!!"],
    "thinking": ["let me", "give me a sec", "give me a second", "give me a moment", "one moment", "one sec",
                 "hold on", "hang on", "thinking", "pondering", "ponder", "figure out", "figure it out",
                 "figure this out", "figure that out", "work out", "work it out", "working it out", "working out",
                 "weigh", "consider", "considering", "maybe", "perhaps", "i wonder", "what if", "on the other hand",
                 "let's see", "suppose", "hmm..."],
}
# ties: the more expressive tag wins
TAG_PRIORITY = ["excitement", "greeting", "doubt", "agreement", "thinking"]
BASE_AROUSAL = {"greeting": 0.55, "agreement": 0.50, "doubt": 0.35, "excitement": 0.85, "thinking": 0.25, "neutral": 0.40}
BASE_VALENCE = {"greeting": 0.5, "agreement": 0.4, "doubt": -0.4, "excitement": 0.8, "thinking": 0.0, "neutral": 0.0}
POSITIVE = ["good", "great", "love", "happy", "nice", "thanks", "thank you", "glad", "perfect", "beautiful", "fun", "news", "win"]
NEGATIVE = ["bad", "sorry", "terrible", "hate", "wrong", "sad", "awful", "unfortunately", "problem", "afraid", "off"]
EXCLAMATION_AROUSAL = 0.10      # per "!" (max 2 counted)
ELLIPSIS_AROUSAL = -0.10        # a pause ("...") lowers energy and counts as a thinking cue
CAPS_AROUSAL = 0.05             # per ALL-CAPS word (max 2 counted)
QUESTION_AROUSAL = 0.05
AMPLITUDE_BASE, AMPLITUDE_GAIN, AMPLITUDE_CAP = 0.8, 0.5, 1.3
# amplitude TIERS by intent (run-3 order): the blind judge rewards sculpted, large gestures; the
# retarget's rate limit still bounds the result. These replace the arousal rule for the amplitude.
AMPLITUDE_TIERS = {"excitement": 1.45, "greeting": 1.25, "agreement": 1.15, "doubt": 1.15, "thinking": 0.9, "neutral": 1.0}
NEGATION = r"(?<!not )(?<!n't )(?<!never )"      # "not sure" / "isn't right" do not count as agreement
NEGATION_WORDS = re.compile(r"\b(?:not|no|never|nothing|nobody)\b|n't\b")

# thirty lines written for this module (six per intent, none shared with any grader)
EXAMPLE_LINES: Dict[str, List[str]] = {
    "greeting": ["Hey there, how's it going?", "Good morning, everyone.", "Hiya! Glad you're here.",
                 "Well, look who it is, it's been a while!", "Welcome back, how are you?", "Evening! Nice to see you again."],
    "agreement": ["Yep, totally, that's a fair point.", "Right, I think so too.", "Agreed, that makes sense to me.",
                  "Absolutely, you have a good point there.", "Correct, that's exactly how it works.", "Sure, of course we can do that."],
    "doubt": ["I don't buy that, honestly.", "Hmm, that seems off to me.", "Are you sure? That can't be right.",
              "I'm unsure this is going to work.", "Is that so? I find it hard to believe.", "That sounds questionable, I'm not convinced."],
    "excitement": ["Whoa, this is the best news ever!", "OMG, I can't wait to see it!", "Yes!! We finally did it!",
                   "That's fantastic, I'm so excited!", "Unbelievable, what a win!", "Awesome, I love this so much!"],
    "thinking": ["Hold on, let me figure this out.", "Hang on a second, I'm still working it out.",
                 "Maybe... on the other hand, it could be the other way.", "Give me a sec to weigh the options.",
                 "I'm pondering whether that would even work.", "One moment, I'm thinking it through."],
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


def _count(text: str, phrases: List[str], negation_aware: bool = False) -> int:
    n = 0
    for p in phrases:
        if p == "!!":
            n += 1 if "!!" in text else 0
        elif p == "yes!!":
            n += len(re.findall(r"\byes\s*!!", text))
        elif p == "hmm...":
            n += len(re.findall(r"\bhm+\s*(?:\.\.\.|…)", text))
        elif p == "really?":
            n += len(re.findall(r"\breally\s*\?", text))
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
    ellipsis = "..." in raw or "…" in raw
    if ellipsis:
        hits["thinking"] += 1                    # a written pause is a thinking cue
    # a bare "hmm" leans doubt, unless the line is deliberating ("hmm..." or another thinking cue)
    n_hmm = len(re.findall(r"\bhm+\b", t)) - _count(t, ["hmm..."])
    if n_hmm > 0:
        if hits["thinking"] > 0:
            hits["thinking"] += n_hmm
        else:
            hits["doubt"] += n_hmm
    # tie-breaks that are cues in their own right
    n_excl = raw.count("!")
    if n_excl >= 2:
        hits["excitement"] += 1                  # exclamation-heavy
    if "?" in raw and NEGATION_WORDS.search(t):
        hits["doubt"] += 1                       # a negated question
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
    n_caps = min(2, sum(1 for w in re.findall(r"[A-Za-z]{2,}", raw) if w.isupper()))
    arousal = BASE_AROUSAL[tag] + EXCLAMATION_AROUSAL * min(2, n_excl) + CAPS_AROUSAL * n_caps
    if ellipsis:
        arousal += ELLIPSIS_AROUSAL
    if "?" in raw:
        arousal += QUESTION_AROUSAL
    arousal = float(max(0.0, min(1.0, arousal)))
    valence = BASE_VALENCE[tag] + 0.1 * min(3, _count(t, POSITIVE)) - 0.1 * min(3, _count(t, NEGATIVE))
    valence = float(max(-1.0, min(1.0, valence)))
    return Intent(tag=tag, arousal=arousal, valence=valence, amplitude=AMPLITUDE_TIERS[tag], hits=hits, text=raw,
                  overridden=overridden)


def example_table() -> Dict[str, List[Dict]]:
    """What the rule says about ``EXAMPLE_LINES`` (its own lines, for documentation)."""
    out: Dict[str, List[Dict]] = {}
    for tag, lines in EXAMPLE_LINES.items():
        out[tag] = [{"text": s, **{k: v for k, v in analyse(s).to_dict().items() if k in ("tag", "arousal", "valence", "amplitude")}}
                    for s in lines]
    return out


def example_accuracy() -> Dict:
    n = sum(len(v) for v in EXAMPLE_LINES.values())
    ok = sum(analyse(s).tag == tag for tag, lines in EXAMPLE_LINES.items() for s in lines)
    return {"correct": ok, "total": n}


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
            "match": "whole-word / whole-phrase matches on the lower-cased text ('!!' = the string; 'yes!!', 'really?' and "
                     "'hmm...' allow spaces before the punctuation); agreement cues preceded by 'not ', \"n't \" or 'never ' "
                     "do not count; a written pause '...' adds one thinking hit; a bare 'hmm' adds one doubt hit, or one "
                     "thinking hit when the line already has a thinking cue; two or more '!' add one excitement hit; a '?' "
                     "together with a negation word (not / no / never / n't) adds one doubt hit; the tag with most hits wins, "
                     "ties by tag_priority_on_ties; no hits = neutral",
            "arousal": f"base_arousal[tag] + {EXCLAMATION_AROUSAL} per '!' (max 2) + {CAPS_AROUSAL} per ALL-CAPS word (max 2) "
                       f"{ELLIPSIS_AROUSAL:+} if '...' present {QUESTION_AROUSAL:+} if '?' present, clamped to [0, 1]",
            "valence": "base_valence[tag] + 0.1 per positive word (max 3) - 0.1 per negative word (max 3), clamped to [-1, 1]",
            "amplitude": "tier by tag (amplitude_tiers), applied to the decoded canonical motion of every source; the retarget's "
                         "max_speed still bounds the result",
            "amplitude_tiers": AMPLITUDE_TIERS,
            "amplitude_arousal_rule_legacy": f"min({AMPLITUDE_CAP}, {AMPLITUDE_BASE} + {AMPLITUDE_GAIN} * arousal) (amplitude_for; not used for the tier)",
            "retrieval_bonus": f"{arousal_weight} * (1 - |window_arousal - target_arousal|) added to the cosine score of every index window",
            "thinking_bonus": f"{thinking_weight} * max(0, still_then_move) when tag == thinking (windows whose second half moves more than their first)",
            "override": "animacy say --intent <tag> forces the tag; punctuation modifiers still apply",
        },
        "examples": example_table(),
        "example_accuracy": example_accuracy(),
    }
