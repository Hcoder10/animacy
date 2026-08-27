// Intent conditioning: text → (tag, arousal, valence, amplitude). Port of
// animacy/model/intent.py `analyse`, driven by the `intent` block of
// web/models/model.json (lexicon, base arousal/valence, cue lists) so the
// browser produces the same numbers as `animacy say`. The block is optional:
// without it the lexicon below (intent.v2, verbatim) is used.
//
// Rules (docs: model.json intent.rules):
//   hits[tag] = whole-word / whole-phrase matches on the lower-cased text
//               ('!!' = the literal string; 'really?' and 'hmm...' allow spaces
//               before the punctuation); agreement cues preceded by "not ",
//               "n't " or "never " do not count; a written pause '...' / '…'
//               adds one thinking hit; most hits wins, ties by tag_priority;
//               no hits → neutral; an override forces the tag
//   arousal   = base_arousal[tag] + 0.10·min(2, #'!') + 0.05·min(2, #ALL-CAPS words)
//               − 0.10 if '...' present + 0.05 if '?' present, clamped to [0, 1]
//   valence   = base_valence[tag] + 0.1·min(3, #positive) − 0.1·min(3, #negative), clamped to [−1, 1]
//   amplitude = min(1.3, 0.8 + 0.5·arousal)

export const TAGS = ['greeting', 'agreement', 'doubt', 'excitement', 'thinking', 'neutral'];

const DEFAULT_INTENT = {
  lexicon_version: 'intent.v2',
  tags: TAGS,
  lexicon: {
    greeting: ['hi', 'hey', 'hello', 'howdy', 'welcome', 'good morning', 'good afternoon', 'good evening', 'good to see', 'nice to meet', 'nice to see', 'great to see', 'greetings', 'long time'],
    agreement: ['yes', 'yeah', 'yep', 'exactly', 'right', 'agree', 'agreed', 'of course', 'sure', 'absolutely', 'correct', 'indeed', 'definitely', 'makes sense', 'precisely', 'true', 'fair enough'],
    doubt: ['no', 'nope', 'nah', 'not sure', 'not true', "don't think", 'do not think', 'doubt', 'disagree', 'not really', 'really?', 'are you sure', 'hardly', 'unlikely', 'not convinced', 'wrong', "i don't know", 'hmm'],
    excitement: ['wow', 'no way', 'incredible', 'amazing', 'awesome', 'fantastic', 'unbelievable', "can't believe", 'cannot believe', 'brilliant', 'wonderful', 'excellent', 'so cool', 'love it', 'yay', 'congratulations', 'oh my', "let's go", '!!'],
    thinking: ['let me think', 'let me see', "let's see", 'wait', 'consider', 'i wonder', 'what if', 'suppose', 'hold on', 'give me a moment', 'thinking', 'maybe', 'perhaps', 'hmm...'],
  },
  tag_priority_on_ties: ['excitement', 'greeting', 'doubt', 'agreement', 'thinking'],
  base_arousal: { greeting: 0.55, agreement: 0.5, doubt: 0.35, excitement: 0.85, thinking: 0.25, neutral: 0.4 },
  base_valence: { greeting: 0.5, agreement: 0.4, doubt: -0.4, excitement: 0.8, thinking: 0.0, neutral: 0.0 },
  positive: ['good', 'great', 'love', 'happy', 'nice', 'thanks', 'thank you', 'glad', 'perfect', 'beautiful', 'fun', 'news'],
  negative: ['bad', 'sorry', 'terrible', 'hate', 'wrong', 'sad', 'awful', 'unfortunately', 'problem', 'afraid'],
};

const EXCLAMATION_AROUSAL = 0.10;
const ELLIPSIS_AROUSAL = -0.10;
const CAPS_AROUSAL = 0.05;
const QUESTION_AROUSAL = 0.05;
const AMPLITUDE_BASE = 0.8, AMPLITUDE_GAIN = 0.5, AMPLITUDE_CAP = 1.3;
const NEGATION = "(?<!not )(?<!n't )(?<!never )";

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const countAll = (text, re) => (text.match(re) || []).length;

function countPhrases(text, phrases, negationAware = false) {
  let n = 0;
  for (const p of phrases) {
    if (p === '!!') n += text.includes('!!') ? 1 : 0;
    else if (p === 'hmm...') n += countAll(text, /hmm+\s*(?:\.\.\.|…)/g);
    else if (p === 'really?') n += countAll(text, /really\s*\?/g);
    else n += countAll(text, new RegExp(`${negationAware ? NEGATION : ''}(?<![a-z'])${escapeRe(p)}(?![a-z'])`, 'g'));
  }
  return n;
}

/** The amplitude rule: 0.8 at arousal 0, 1.3 at arousal 1 (capped). */
export function amplitudeFor(arousal) {
  return Math.min(AMPLITUDE_CAP, AMPLITUDE_BASE + AMPLITUDE_GAIN * Math.max(0, Math.min(1, Number(arousal) || 0)));
}

/**
 * @param {string} text
 * @param {object} [o]
 * @param {string|null} [o.override]  force the tag (punctuation modifiers still apply)
 * @param {object|null} [o.spec]      model.json `intent` block; default = the built-in intent.v2 table
 * @returns {{tag:string, arousal:number, valence:number, amplitude:number, hits:object, text:string, overridden:boolean}}
 */
export function analyse(text, { override = null, spec = null } = {}) {
  const S = spec && spec.lexicon ? spec : DEFAULT_INTENT;
  const raw = text || '';
  const t = raw.toLowerCase().trim();
  const hits = {};
  for (const tag of Object.keys(S.lexicon)) hits[tag] = countPhrases(t, S.lexicon[tag], tag === 'agreement');
  if (raw.includes('...') || raw.includes('…')) hits.thinking = (hits.thinking || 0) + 1;
  let tag, overridden;
  if (override) {
    if (!(S.tags || TAGS).includes(override)) throw new Error(`unknown intent '${override}'; choose from ${(S.tags || TAGS).join(', ')}`);
    tag = override;
    overridden = true;
  } else {
    const best = Math.max(0, ...Object.values(hits));
    tag = 'neutral';
    if (best > 0) for (const cand of S.tag_priority_on_ties) if (hits[cand] === best) { tag = cand; break; }
    overridden = false;
  }
  const nExcl = Math.min(2, countAll(raw, /!/g));
  const nCaps = Math.min(2, (raw.match(/[A-Za-z]{2,}/g) || []).filter((w) => w === w.toUpperCase()).length);
  let arousal = S.base_arousal[tag] + EXCLAMATION_AROUSAL * nExcl + CAPS_AROUSAL * nCaps;
  if (raw.includes('...') || raw.includes('…')) arousal += ELLIPSIS_AROUSAL;
  if (raw.includes('?')) arousal += QUESTION_AROUSAL;
  arousal = Math.max(0, Math.min(1, arousal));
  let valence = S.base_valence[tag] + 0.1 * Math.min(3, countPhrases(t, S.positive)) - 0.1 * Math.min(3, countPhrases(t, S.negative));
  valence = Math.max(-1, Math.min(1, valence));
  return { tag, arousal, valence, amplitude: amplitudeFor(arousal), hits, text: raw, overridden };
}
