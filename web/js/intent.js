// Intent conditioning: text → (tag, arousal, valence, amplitude). Port of
// animacy/model/intent.py `analyse` (intent.v3), driven by the `intent` block of
// web/models/model.json (lexicon, base arousal/valence, cue lists, amplitude
// tiers) so the browser produces the same numbers as `animacy say`. The block
// is optional: without it the lexicon below (intent.v3, verbatim) is used.
//
// Rules (docs: model.json intent.rules):
//   hits[tag] = whole-word / whole-phrase matches on the lower-cased text
//               ('!!' = the literal string; 'yes!!', 'really?' and 'hmm...' allow
//               spaces before the punctuation); agreement cues preceded by "not ",
//               "n't " or "never " do not count; a written pause '...' / '…' adds one
//               thinking hit; a bare "hmm" counts as thinking when the line already
//               has a thinking cue, else as doubt; two or more '!' add one excitement
//               hit; a '?' together with a negation word adds one doubt hit; most hits
//               wins, ties by tag_priority; no hits → neutral; an override forces the tag
//   arousal   = base_arousal[tag] + 0.10·min(2, #'!') + 0.05·min(2, #ALL-CAPS words)
//               − 0.10 if '...' present + 0.05 if '?' present, clamped to [0, 1]
//   valence   = base_valence[tag] + 0.1·min(3, #positive) − 0.1·min(3, #negative), clamped to [−1, 1]
//   amplitude = amplitude_tiers[tag]   (excitement 1.45 … thinking 0.9; the arousal rule
//               min(1.3, 0.8 + 0.5·arousal) is kept as `amplitudeFor` for reference only)

export const TAGS = ['greeting', 'agreement', 'doubt', 'excitement', 'thinking', 'neutral'];

export const AMPLITUDE_TIERS = { excitement: 1.45, greeting: 1.25, agreement: 1.15, doubt: 1.15, thinking: 0.9, neutral: 1.0 };

const DEFAULT_INTENT = {
  lexicon_version: 'intent.v3',
  tags: TAGS,
  lexicon: {
    greeting: ['hi', 'hey', 'hello', 'hiya', 'yo', 'howdy', 'hey there', 'hi there', 'morning', 'afternoon', 'evening',
      'good morning', 'good afternoon', 'good evening', 'good day', 'welcome', 'welcome back',
      'nice to see', 'great to see', 'lovely to see', 'nice to meet', 'pleased to meet', "glad you're here",
      'glad you are here', 'there you are', "it's been a while", 'it has been a while', 'long time',
      'how are you', "how's it going", 'how is it going', 'how have you been', 'greetings'],
    agreement: ['yes', 'yeah', 'yep', 'yup', 'right', 'exactly', 'agreed', 'agree', 'true', 'correct', 'sure',
      'of course', 'absolutely', 'definitely', 'totally', 'indeed', 'precisely', 'makes sense',
      'fair point', 'good point', 'fair enough', 'i think so', "you're right", 'you are right',
      'well said', 'no doubt', "that's it"],
    doubt: ['no', 'nope', 'nah', 'not sure', 'unsure', 'not true', 'doubt', 'doubtful', "i don't buy", "don't buy",
      "don't believe", 'do not believe', 'hard to believe', 'seems off', 'sounds off', "that can't be",
      "can't be right", 'not convinced', 'questionable', 'really?', 'is that so', 'are you sure',
      "don't think", 'do not think', 'disagree', 'not really', 'hardly', 'unlikely', 'wrong', "i don't know",
      'not so sure', 'skeptical', 'sceptical', 'i doubt', 'not buying'],
    excitement: ['wow', 'whoa', 'omg', 'oh my', 'no way', 'incredible', 'amazing', 'awesome', 'fantastic',
      'unbelievable', 'best', 'love', "can't wait", 'cannot wait', 'so excited', 'yes!!', "let's go",
      'brilliant', 'wonderful', 'excellent', 'so cool', 'yay', 'congratulations', 'congrats',
      'what a win', 'thrilled', "can't believe", 'cannot believe', '!!'],
    thinking: ['let me', 'give me a sec', 'give me a second', 'give me a moment', 'one moment', 'one sec',
      'hold on', 'hang on', 'thinking', 'pondering', 'ponder', 'figure out', 'figure it out',
      'figure this out', 'figure that out', 'work out', 'work it out', 'working it out', 'working out',
      'weigh', 'consider', 'considering', 'maybe', 'perhaps', 'i wonder', 'what if', 'on the other hand',
      "let's see", 'suppose', 'hmm...'],
  },
  tag_priority_on_ties: ['excitement', 'greeting', 'doubt', 'agreement', 'thinking'],
  base_arousal: { greeting: 0.55, agreement: 0.5, doubt: 0.35, excitement: 0.85, thinking: 0.25, neutral: 0.4 },
  base_valence: { greeting: 0.5, agreement: 0.4, doubt: -0.4, excitement: 0.8, thinking: 0.0, neutral: 0.0 },
  positive: ['good', 'great', 'love', 'happy', 'nice', 'thanks', 'thank you', 'glad', 'perfect', 'beautiful', 'fun', 'news', 'win'],
  negative: ['bad', 'sorry', 'terrible', 'hate', 'wrong', 'sad', 'awful', 'unfortunately', 'problem', 'afraid', 'off'],
  rules: { amplitude_tiers: AMPLITUDE_TIERS },
};

const EXCLAMATION_AROUSAL = 0.10;
const ELLIPSIS_AROUSAL = -0.10;
const CAPS_AROUSAL = 0.05;
const QUESTION_AROUSAL = 0.05;
const AMPLITUDE_BASE = 0.8, AMPLITUDE_GAIN = 0.5, AMPLITUDE_CAP = 1.3;
const NEGATION = "(?<!not )(?<!n't )(?<!never )";
const NEGATION_WORDS = /\b(?:not|no|never|nothing|nobody)\b|n't\b/;

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const countAll = (text, re) => (text.match(re) || []).length;

function countPhrases(text, phrases, negationAware = false) {
  let n = 0;
  for (const p of phrases) {
    if (p === '!!') n += text.includes('!!') ? 1 : 0;
    else if (p === 'yes!!') n += countAll(text, /\byes\s*!!/g);
    else if (p === 'hmm...') n += countAll(text, /\bhm+\s*(?:\.\.\.|…)/g);
    else if (p === 'really?') n += countAll(text, /\breally\s*\?/g);
    else n += countAll(text, new RegExp(`${negationAware ? NEGATION : ''}(?<![a-z'])${escapeRe(p)}(?![a-z'])`, 'g'));
  }
  return n;
}

/** The legacy arousal→amplitude rule (intent.amplitude_for); the tiers are what ships. */
export function amplitudeFor(arousal) {
  return Math.min(AMPLITUDE_CAP, AMPLITUDE_BASE + AMPLITUDE_GAIN * Math.max(0, Math.min(1, Number(arousal) || 0)));
}

function tiersOf(S) {
  return (S.rules && S.rules.amplitude_tiers) || S.amplitude_tiers || AMPLITUDE_TIERS;
}

/**
 * @param {string} text
 * @param {object} [o]
 * @param {string|null} [o.override]  force the tag (punctuation modifiers still apply)
 * @param {object|null} [o.spec]      model.json `intent` block; default = the built-in intent.v3 table
 * @returns {{tag:string, arousal:number, valence:number, amplitude:number, hits:object, text:string, overridden:boolean}}
 */
export function analyse(text, { override = null, spec = null } = {}) {
  const S = spec && spec.lexicon ? spec : DEFAULT_INTENT;
  const raw = text || '';
  const t = raw.toLowerCase().trim();
  const hits = {};
  for (const tag of Object.keys(S.lexicon)) hits[tag] = countPhrases(t, S.lexicon[tag], tag === 'agreement');
  const ellipsis = raw.includes('...') || raw.includes('…');
  if (ellipsis) hits.thinking = (hits.thinking || 0) + 1;              // a written pause is a thinking cue
  // a bare "hmm" leans doubt, unless the line is deliberating ("hmm..." or another thinking cue)
  const nHmm = countAll(t, /\bhm+\b/g) - countPhrases(t, ['hmm...']);
  if (nHmm > 0) {
    if (hits.thinking > 0) hits.thinking += nHmm;
    else hits.doubt = (hits.doubt || 0) + nHmm;
  }
  // tie-breaks that are cues in their own right
  const nExcl = countAll(raw, /!/g);
  if (nExcl >= 2) hits.excitement = (hits.excitement || 0) + 1;         // exclamation-heavy
  if (raw.includes('?') && NEGATION_WORDS.test(t)) hits.doubt = (hits.doubt || 0) + 1; // a negated question
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
  const nCaps = Math.min(2, (raw.match(/[A-Za-z]{2,}/g) || []).filter((w) => w === w.toUpperCase()).length);
  let arousal = S.base_arousal[tag] + EXCLAMATION_AROUSAL * Math.min(2, nExcl) + CAPS_AROUSAL * nCaps;
  if (ellipsis) arousal += ELLIPSIS_AROUSAL;
  if (raw.includes('?')) arousal += QUESTION_AROUSAL;
  arousal = Math.max(0, Math.min(1, arousal));
  let valence = S.base_valence[tag] + 0.1 * Math.min(3, countPhrases(t, S.positive)) - 0.1 * Math.min(3, countPhrases(t, S.negative));
  valence = Math.max(-1, Math.min(1, valence));
  const tiers = tiersOf(S);
  return { tag, arousal, valence, amplitude: tiers[tag] ?? AMPLITUDE_TIERS[tag] ?? 1.0, hits, text: raw, overridden };
}
