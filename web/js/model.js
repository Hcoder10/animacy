// The motion model in the browser: audio features → codes → canonical motion.
//
// Reproduces animacy/model/infer.py `generate_motion` on the bundle written by
// animacy/model/export.py into web/models/:
//   a2m.onnx        features [1, L, 66] f32, speaking [1, L] i64, causal [1] i64 → logits [1, L, 512]
//   vq_decoder.onnx codes [1, L] i64 → motion [1, 2L, 14] f32 (canonical units, de-standardised in-graph)
//   bigram.bin      float16 [512, 512] log P(next | prev), row = prev
//   model.json      channel order, sampling defaults, smoothing cutoff
// plus the retrieval (motion-matching) index retrieval.json + retrieval.bin, and
// the speech-envelope heuristic from animacy/serve.py as the no-model fallback.
//
// Two archs (model.json `archs` / `default_arch`; v1 files without them are "ff"):
//   ff  a2m.onnx     one pass → logits [L,512]; infer.sample_codes: softmax(logits/T + w·bigram[prev])
//   ar  a2m_ar.onnx  stepped per code with the history [BOS, …] → logits_next; a2m_ar.generate:
//                    temperature, top-p, repeat penalty (+ stay bias on quiet audio)
// The RNG is sfc32 (not numpy's PCG64), so a seeded run is reproducible in the
// browser but not bit-identical to Python; with T → 0 both reduce to argmax and
// tests/test_web_model_parity.py checks that path exactly for both archs.

import { float16ToFloat32, seededRng, smoothColumns, hanning, convolveSame, butter2, filtfilt } from './dsp.js';
import { N_FEATS, RATE_HZ } from './features.js';
import { neutralFrame, BOUNDS } from './canonical.js';

// ---------------------------------------------------------------------------
// generation-side post-processing — infer.postprocess_motion, applied to every
// source (model, retrieval) in the order model.json `postprocess.order`:
//   amplitude (per-channel scale) → pitch floor → utterance-final settle → (clamp in motionToFrames)
// ---------------------------------------------------------------------------
export const BASELINE_HZ = 0.3;   // the cutoff the pose channels were detrended with
export const QUIET_ENERGY = -0.3; // normalised log energy below this = silence (feature 64)

/**
 * @param {Float32Array} motion  flat [T*C] canonical units (copied, not modified)
 * @param {number} T
 * @param {string[]} channels    channel order (needs 'head_pitch' for the floor)
 * @param {object} o
 * @param {Uint8Array|number[]|null} [o.speaking]  [T]
 * @param {Float32Array[]|null} [o.featRows]        [T][66]
 * @param {number} [o.settleS]        seconds of end-of-utterance blend to neutral (0 = off)
 * @param {number|null} [o.pitchFloor] degrees the 0.3 Hz head_pitch baseline may not go below
 * @param {number|number[]} [o.amplitude]  scalar or per-channel scale
 */
export function postprocessMotion(motion, T, channels, { speaking = null, featRows = null, settleS = 0, pitchFloor = null, amplitude = 1.0, rateHz = RATE_HZ } = {}) {
  const C = channels.length;
  const m = Float32Array.from(motion);
  const amp = Array.isArray(amplitude) ? amplitude : new Array(C).fill(Number(amplitude));
  if (amp.some((a) => a !== 1.0)) for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) m[t * C + c] *= amp[c];
  const pi = channels.indexOf('head_pitch');
  if (pitchFloor !== null && pitchFloor !== undefined && T > 0 && pi >= 0) {
    const p = new Float64Array(T);
    for (let t = 0; t < T; t++) p[t] = m[t * C + pi];
    let base;
    if (T >= 12) {
      const { b, a } = butter2(BASELINE_HZ, rateHz);
      base = filtfilt(b, a, p, Math.min(9, T - 1));
    } else {
      let s = 0; for (let t = 0; t < T; t++) s += p[t];
      base = new Float64Array(T).fill(s / T);
    }
    for (let t = 0; t < T; t++) m[t * C + pi] = p[t] + Math.max(0.0, pitchFloor - base[t]);
  }
  if (settleS && settleS > 0 && T > 0) {
    const n = Math.max(1, Math.round(settleS * rateHz));
    let end = T;
    if (speaking && speaking.length === T && Array.from(speaking).some((v) => v > 0)) {
      for (let t = T - 1; t >= 0; t--) if (speaking[t] > 0) { end = t + 1; break; }
    } else if (featRows && featRows.length === T && featRows[0].length > 64) {
      end = 0;
      for (let t = T - 1; t >= 0; t--) if (featRows[t][64] > QUIET_ENERGY) { end = t + 1; break; }
      if (end === 0) end = T;
    }
    const w = new Float32Array(T);
    const a0 = Math.max(0, end - n);
    // np.linspace(0, 1, n+1)[1:] = k/n for k = 1..n; keep the last (end − a0) of them
    for (let t = a0; t < end; t++) w[t] = (n - (end - t) + 1) / n;
    for (let t = end; t < T; t++) w[t] = 1.0;
    for (let t = 0; t < T; t++) { const g = 1.0 - w[t]; for (let c = 0; c < C; c++) m[t * C + c] *= g; }
  }
  return m;
}

export const ORT_VERSION = '1.20.1';
export const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.mjs`;
export const ORT_WASM_DIR = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const FRAMES_PER_CODE = 2;

let _ortPromise = null;
/** onnxruntime-web, loaded once from the CDN. wasm EP, single-threaded (Pages is not cross-origin isolated). */
export function loadOrt() {
  if (!_ortPromise) {
    _ortPromise = import(ORT_URL).then((ort) => {
      ort.env.wasm.wasmPaths = ORT_WASM_DIR;
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';
      return ort;
    });
  }
  return _ortPromise;
}

/** fetch → Uint8Array with byte progress (Content-Length permitting). Throws on !ok. */
export async function fetchBytes(url, onProgress = null) {
  const r = await fetch(url, { cache: 'force-cache' });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !onProgress) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

// ---------------------------------------------------------------------------
// windows / sampling (animacy.model.data / infer)
// ---------------------------------------------------------------------------
/** 30 Hz rows [T][dim] → 15 Hz flat Float32Array [L*dim] by averaging pairs (odd tail dropped). */
export function poolPairs(rows, dim) {
  const L = Math.floor(rows.length / FRAMES_PER_CODE);
  const out = new Float32Array(L * dim);
  for (let l = 0; l < L; l++) {
    const a = rows[2 * l], b = rows[2 * l + 1];
    for (let d = 0; d < dim; d++) out[l * dim + d] = 0.5 * (a[d] + b[d]);
  }
  return out;
}

/** 30 Hz 0/1 flags → 15 Hz BigInt64Array (any of the pair). */
export function poolFlag(flags) {
  const L = Math.floor(flags.length / FRAMES_PER_CODE);
  const out = new BigInt64Array(L);
  for (let l = 0; l < L; l++) out[l] = flags[2 * l] || flags[2 * l + 1] ? 1n : 0n;
  return out;
}

/**
 * [L, n] logits → Int32Array codes (infer.sample_codes).
 * @param {Float32Array} logits  flat [L*n]
 * @param {Float32Array|null} bigram  flat [n*n] log P(next|prev), row = prev
 */
export function sampleCodes(logits, L, n, bigram, { temperature = 0.8, bigramWeight = 0.5, seed = 0 } = {}) {
  const rng = seededRng(seed);
  const codes = new Int32Array(L);
  const invT = 1.0 / Math.max(temperature, 1e-6);
  const z = new Float64Array(n);
  let prev = -1;
  for (let t = 0; t < L; t++) {
    let zmax = -Infinity;
    for (let k = 0; k < n; k++) {
      let v = logits[t * n + k] * invT;
      if (prev >= 0 && bigram && bigramWeight > 0) v += bigramWeight * bigram[prev * n + k];
      z[k] = v;
      if (v > zmax) zmax = v;
    }
    let sum = 0;
    for (let k = 0; k < n; k++) { z[k] = Math.exp(z[k] - zmax); sum += z[k]; }
    const u = rng() * sum;
    let acc = 0, pick = n - 1;
    for (let k = 0; k < n; k++) { acc += z[k]; if (u < acc) { pick = k; break; } }
    prev = pick;
    codes[t] = pick;
  }
  return codes;
}

// ---------------------------------------------------------------------------
// canonical frames from model channels
// ---------------------------------------------------------------------------
/**
 * [T, C] motion in `channels` order → canonical frames (neutral elsewhere,
 * face_valid = 1, arm_valid = 0, clipped to the schema bounds) — infer.motion_to_clip.
 */
export function motionToFrames(motion, T, channels, speaking = null, rateHz = RATE_HZ) {
  const C = channels.length;
  const frames = [];
  for (let t = 0; t < T; t++) {
    const f = neutralFrame();
    f.t = t / rateHz;
    for (let c = 0; c < C; c++) {
      const [lo, hi] = BOUNDS[channels[c]] || [-Infinity, Infinity];
      f[channels[c]] = Math.min(Math.max(motion[t * C + c], lo), hi);
    }
    f.face_valid = 1;
    f.arm_valid = 0;
    if (speaking) f.speaking = speaking[Math.min(t, speaking.length - 1)] ? 1 : 0;
    frames.push(f);
  }
  return frames;
}

/**
 * One categorical draw with temperature, nucleus (top-p) and an optional penalty
 * on repeating `prev` — a2m_ar.AudioToMotionAR.sample, line for line
 * (np.searchsorted(cum, top_p) keeps every sorted entry up to and including the
 * first whose cumulative mass reaches top_p).
 */
export function sampleTopP(logits, n, temperature, topP, rng, prev = null, repeatPenalty = 0.0) {
  const z = new Float64Array(n);
  for (let k = 0; k < n; k++) z[k] = logits[k];
  if (prev !== null && prev >= 0 && prev < n && repeatPenalty) z[prev] -= repeatPenalty;
  const invT = 1.0 / Math.max(temperature, 1e-6);
  let zmax = -Infinity;
  for (let k = 0; k < n; k++) { z[k] *= invT; if (z[k] > zmax) zmax = z[k]; }
  let sum = 0;
  for (let k = 0; k < n; k++) { z[k] = Math.exp(z[k] - zmax); sum += z[k]; }
  for (let k = 0; k < n; k++) z[k] /= sum;
  if (topP > 0 && topP < 1) {
    const order = Array.from({ length: n }, (_, k) => k).sort((a, b) => z[b] - z[a]);
    let cum = 0, keepMass = 0, i = 0;
    for (; i < n; i++) { cum += z[order[i]]; keepMass += z[order[i]]; if (cum >= topP) break; }
    const keep = new Set(order.slice(0, Math.min(i + 1, n)));
    for (let k = 0; k < n; k++) z[k] = keep.has(k) ? z[k] / keepMass : 0;
  }
  const u = rng();
  let acc = 0, pick = n - 1;
  for (let k = 0; k < n; k++) { acc += z[k]; if (u < acc) { pick = k; break; } }
  return pick;
}

// ---------------------------------------------------------------------------
// code samplers, keyed by model.json arch ("ff" | "ar"; v1 files without an
// `archs` key are "ff")
// ---------------------------------------------------------------------------
// A sampler turns 15 Hz features into codes. Signature:
//   async (model, feat15: Float32Array [L*66], L, speaking15: BigInt64Array [L],
//          opts) → Int32Array codes [L]
//
// "ff": one non-autoregressive pass (a2m.onnx) gives logits for every step,
//   then the bigram-prior categorical draw of infer.sample_codes.
// "ar": a2m_ar.onnx is stepped once per code with the whole history
//   [BOS, c0, …] as input (exact: the decoder's self-attention window is
//   `window` codes per layer) → logits_next; temperature / top-p / repeat
//   penalty (+ optional stay bias on quiet audio) per a2m_ar.generate.
export const SAMPLERS = {
  async ff(model, feat15, L, speaking15, { causal, temperature, bigramWeight, seed }) {
    const lg = await model.logits(feat15, L, speaking15, causal);
    return sampleCodes(lg, L, model.nCodes, model.bigram, { temperature, bigramWeight, seed });
  },
  async ar(model, feat15, L, speaking15, { causal, temperature, topP, repeatPenalty, seed, stayBias = 0, stayEnergy = -0.3, onStep = null }) {
    const n = model.nCodes;
    const bos = (model.meta.a2m_ar && model.meta.a2m_ar.bos) ?? n;
    const rng = seededRng(seed);
    const hist = [bos];
    const out = new Int32Array(L);
    for (let t = 0; t < L; t++) {
      const logits = await model.logitsNext(feat15, L, speaking15, causal, hist);
      const prev = t > 0 ? out[t - 1] : null;
      if (prev !== null && stayBias && feat15[t * N_FEATS + 64] < stayEnergy) logits[prev] += stayBias;
      const c = sampleTopP(logits, n, temperature, topP, rng, prev, repeatPenalty);
      out[t] = c;
      hist.push(c);
      if (onStep) onStep(t + 1, L);
    }
    return out;
  },
};
const ARCH_ALIASES = { a2m: 'ff' };

/**
 * Which arch a model.json asks for and which this build can run.
 * v1 (no `archs`): "ff" via `a2m`. v2: `archs` + `default_arch`, blocks `a2m`
 * ("ff") and `a2m_ar` ("ar"). An arch we cannot run (no sampler, or its file
 * missing from the json) falls back to the first listed arch we can; none → null.
 */
export function resolveArch(meta) {
  const blocks = { ff: meta.a2m || null, ar: meta.a2m_ar || null };
  const norm = (a) => ARCH_ALIASES[a] || a;
  const listed = (meta.archs && meta.archs.length ? meta.archs : Object.keys(blocks).filter((a) => blocks[a])).map(norm);
  const usable = (a) => !!(SAMPLERS[a] && blocks[a] && blocks[a].file);
  const want = norm(meta.default_arch || meta.arch || listed[0] || 'ff');
  if (usable(want)) return { arch: want, wanted: want, block: blocks[want] };
  const alt = listed.find(usable) || Object.keys(blocks).find(usable);
  return alt ? { arch: alt, wanted: want, block: blocks[alt] } : null;
}

// ---------------------------------------------------------------------------
// the learned model
// ---------------------------------------------------------------------------
export class MotionModel {
  /**
   * @param {object} meta       model.json
   * @param {object} sessions   {ff?: ort session (a2m.onnx), ar?: ort session (a2m_ar.onnx)}
   * @param {object} decoder    vq_decoder.onnx session
   * @param {Float32Array|null} bigram  [n*n] log P(next|prev), ff only
   * @param {string} arch       resolved arch ("ff" | "ar")
   */
  constructor(meta, sessions, decoder, bigram, arch) {
    this.meta = meta;
    this.sessions = sessions;
    this.a2m = sessions.ff || null;     // kept for the parity tests / ff path
    this.decoder = decoder;
    this.bigram = bigram;               // Float32Array [n*n] or null
    this.channels = meta.channels;
    this.nCodes = meta.n_codes;
    this.arch = arch;
    this.sampling = meta.sampling || { temperature: 0.8, bigram_weight: 0.5 };
    this.smoothHz = (meta.smoothing && meta.smoothing.cutoff_hz) || 6.0;
    this.postprocess = meta.postprocess || {};      // {settle_s, pitch_floor, amplitude}
    if (!SAMPLERS[this.arch] || !sessions[this.arch]) throw new Error(`model.json arch '${this.arch}' is not runnable here (samplers: ${Object.keys(SAMPLERS).join(', ')})`);
  }

  /** Human-readable "what will run": arch + file. */
  get describe() {
    const b = this.arch === 'ar' ? this.meta.a2m_ar : this.meta.a2m;
    return `${this.arch}:${b && b.file}`;
  }

  /**
   * @param {string} baseUrl  directory holding model.json (default 'models/')
   * @param {(msg:string, frac:number)=>void} [onProgress]
   */
  static async load(baseUrl = 'models/', onProgress = null) {
    const say = (m, f = 0) => { if (onProgress) onProgress(m, f); };
    say('loading onnxruntime-web…', 0.02);
    const ort = await loadOrt();
    say('reading model.json…', 0.05);
    const mr = await fetch(`${baseUrl}model.json`, { cache: 'no-cache' });
    if (!mr.ok) throw new Error(`no model bundle: ${baseUrl}model.json (${mr.status})`);
    const meta = await mr.json();
    if (!meta.vq_decoder || !meta.vq_decoder.file) throw new Error('model.json has no vq_decoder');
    const res = resolveArch(meta);
    if (!res) throw new Error(`model.json lists no arch this build can run (archs: ${JSON.stringify(meta.archs || Object.keys(meta).filter((k) => k.startsWith('a2m')))})`);
    if (res.arch !== res.wanted) console.warn(`[model] model.json wants arch '${res.wanted}', running '${res.arch}' instead`);
    const files = [res.block.file, meta.vq_decoder.file, res.arch === 'ff' && meta.bigram && meta.bigram.file].filter(Boolean);
    const sizes = [res.block.bytes || 0, meta.vq_decoder.bytes || 0, 0];
    const totalBytes = sizes.reduce((a, b) => a + b, 0) || 1;
    let doneBytes = 0;
    const bufs = [];
    for (let i = 0; i < files.length; i++) {
      const base = doneBytes;
      bufs.push(await fetchBytes(`${baseUrl}${files[i]}`, (got) => say(`downloading ${files[i]} ${(got / 1e6).toFixed(1)} MB`, 0.1 + 0.6 * Math.min(1, (base + got) / totalBytes))));
      doneBytes += sizes[i] || bufs[i].length;
    }
    say('creating sessions…', 0.75);
    const opts = { executionProviders: ['wasm'] };
    const sessions = {};
    sessions[res.arch] = await ort.InferenceSession.create(bufs[0], opts);
    const decoder = await ort.InferenceSession.create(bufs[1], opts);
    let bigram = null;
    if (bufs[2]) {
      const u16 = new Uint16Array(bufs[2].buffer, bufs[2].byteOffset, bufs[2].byteLength >> 1);
      bigram = float16ToFloat32(u16);
      if (bigram.length !== meta.n_codes * meta.n_codes) { console.warn('[model] bigram size mismatch, ignoring prior'); bigram = null; }
    }
    say(`model ready (${res.arch}: ${res.block.file})`, 1);
    return new MotionModel(meta, sessions, decoder, bigram, res.arch);
  }

  /** ff: a2m logits for 15 Hz features. @returns {Promise<Float32Array>} flat [L*n] */
  async logits(feat15, L, speaking15, causal) {
    if (!this.sessions.ff) throw new Error('feed-forward a2m session not loaded');
    const ort = await loadOrt();
    const out = await this.sessions.ff.run({
      features: new ort.Tensor('float32', feat15, [1, L, N_FEATS]),
      speaking: new ort.Tensor('int64', speaking15, [1, L]),
      causal: new ort.Tensor('int64', BigInt64Array.of(causal ? 1n : 0n), [1]),
    });
    return out.logits.data;
  }

  /** ar: logits for the next code given the history [BOS, c0, …]. @returns {Promise<Float32Array>} [n] */
  async logitsNext(feat15, L, speaking15, causal, hist) {
    if (!this.sessions.ar) throw new Error('autoregressive a2m_ar session not loaded');
    const ort = await loadOrt();
    const out = await this.sessions.ar.run({
      features: new ort.Tensor('float32', feat15, [1, L, N_FEATS]),
      speaking: new ort.Tensor('int64', speaking15, [1, L]),
      causal: new ort.Tensor('int64', BigInt64Array.of(causal ? 1n : 0n), [1]),
      codes: new ort.Tensor('int64', BigInt64Array.from(hist, (v) => BigInt(v)), [1, hist.length]),
    });
    return Float32Array.from(out.logits_next.data);
  }

  /** codes [L] → motion flat [2L*C] canonical units */
  async decode(codes) {
    const ort = await loadOrt();
    const c64 = BigInt64Array.from(codes, (v) => BigInt(v));
    const out = await this.decoder.run({ codes: new ort.Tensor('int64', c64, [1, codes.length]) });
    return out.motion.data;
  }

  /**
   * infer.generate_motion: [T][66] features + [T] speaking → {motion: Float32Array [T*C], codes}
   * @param {Float32Array[]} featRows
   * @param {Uint8Array|number[]} speaking
   */
  async generate(featRows, speaking, { causal = false, temperature = null, bigramWeight = null, topP = null, repeatPenalty = null, seed = 0, smoothHz = undefined, onStep = null, amplitude = null, settleS = null, pitchFloor = undefined } = {}) {
    const T = featRows.length;
    const C = this.channels.length;
    const stats = this.meta.stats || {};
    const pp = this.postprocess;
    const ppOpts = {
      speaking, featRows,
      settleS: settleS === null ? (pp.settle_s || 0) : settleS,
      pitchFloor: pitchFloor === undefined ? (pp.pitch_floor === undefined ? null : pp.pitch_floor) : pitchFloor,
      amplitude: amplitude === null ? (pp.amplitude === undefined ? 1.0 : pp.amplitude) : amplitude,
    };
    if (T < FRAMES_PER_CODE) {
      const m = new Float32Array(T * C);
      for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) m[t * C + c] = (stats.mean && stats.mean[c]) || 0;
      return { motion: m, codes: new Int32Array(0), arch: this.arch, postprocess: ppOpts };
    }
    const L = Math.floor(T / FRAMES_PER_CODE);
    const f15 = poolPairs(featRows, N_FEATS);
    const s15 = poolFlag(speaking);
    const arS = (this.meta.a2m_ar && this.meta.a2m_ar.sampling) || {};
    const codes = await SAMPLERS[this.arch](this, f15, L, s15, {
      causal,
      temperature: temperature ?? (this.arch === 'ar' ? arS.temperature : this.sampling.temperature) ?? 0.8,
      bigramWeight: bigramWeight ?? this.sampling.bigram_weight ?? 0.5,
      topP: topP ?? arS.top_p ?? 0.9,
      repeatPenalty: repeatPenalty ?? arS.repeat_penalty ?? 0.0,
      stayBias: arS.stay_bias ?? 0.0,
      stayEnergy: arS.stay_energy ?? -0.3,
      seed,
      onStep,
    });
    let m = await this.decode(codes);                       // [2L, C]
    const T2 = 2 * L;
    const cutoff = smoothHz === undefined ? this.smoothHz : smoothHz;
    m = smoothColumns(m, T2, C, RATE_HZ, cutoff);
    if (T2 < T) {                                            // odd tail tick: hold the last frame
      const padded = new Float32Array(T * C);
      padded.set(m);
      for (let t = T2; t < T; t++) for (let c = 0; c < C; c++) padded[t * C + c] = m[(T2 - 1) * C + c];
      m = padded;
    }
    m = m.slice(0, T * C);
    const amp = Array.isArray(ppOpts.amplitude) ? ppOpts.amplitude.some((a) => a !== 1.0) : ppOpts.amplitude !== 1.0;
    if (ppOpts.settleS || ppOpts.pitchFloor !== null || amp) m = postprocessMotion(m, T, this.channels, ppOpts);
    return { motion: m, codes, arch: this.arch, postprocess: ppOpts };
  }
}

// ---------------------------------------------------------------------------
// retrieval (motion matching) — animacy.model.retrieval.RetrievalIndex
// ---------------------------------------------------------------------------
export class RetrievalIndex {
  constructor(header, keys, motion) {
    this.h = header;
    this.n = header.n;
    this.keyDim = header.key_dim;
    this.win = header.win;
    this.hop = header.hop;
    this.channels = header.channels;
    this.keys = keys;            // Float32Array [n*keyDim]
    this.motion = motion;        // Float32Array [n*win*C]
    this.nextId = Int32Array.from(header.next_id);
    this.speaking = Float32Array.from(header.speaking);
    this.subwindows = header.subwindows;
    this.continuityBonus = header.continuity_bonus ?? 0.1;
    this.speakingBonus = header.speaking_bonus ?? 0.05;
    this.crossfade = header.crossfade ?? 5;
    // intent fields (v2 indexes): per-window human arousal 0..1 and still-then-move −1..1
    this.arousal = Array.isArray(header.arousal) && header.arousal.length === this.n ? Float32Array.from(header.arousal) : null;
    this.stillThenMove = Array.isArray(header.still_then_move) && header.still_then_move.length === this.n ? Float32Array.from(header.still_then_move) : null;
    this.arousalBonus = header.arousal_bonus ?? 0.15;
    this.thinkingBonus = header.thinking_bonus ?? 0.10;
  }

  static async load(baseUrl = 'models/', onProgress = null) {
    const say = (m, f = 0) => { if (onProgress) onProgress(m, f); };
    const hr = await fetch(`${baseUrl}retrieval.json`, { cache: 'no-cache' });
    if (!hr.ok) throw new Error(`no retrieval index: ${baseUrl}retrieval.json (${hr.status})`);
    const h = await hr.json();
    const bin = await fetchBytes(`${baseUrl}${h.bin || 'retrieval.bin'}`, (got, total) => say(`downloading retrieval index ${(got / 1e6).toFixed(1)} MB`, total ? got / total : 0));
    const u16 = new Uint16Array(bin.buffer, bin.byteOffset, bin.byteLength >> 1);
    const all = float16ToFloat32(u16);
    const nk = h.n * h.key_dim;
    const C = h.channels.length;
    const keys = all.slice(0, nk);
    const motion = all.slice(nk, nk + h.n * h.win * C);
    say('retrieval index ready', 1);
    return new RetrievalIndex(h, keys, motion);
  }

  /** [30][66] → L2-normalised key [330] (retrieval.window_key). */
  windowKey(rows, start) {
    const dim = N_FEATS;
    const parts = [[0, this.win], ...this.subwindows];
    const key = new Float32Array(dim * parts.length);
    parts.forEach(([a, b], p) => {
      for (let k = a; k < b; k++) {
        const r = rows[start + k];
        for (let d = 0; d < dim; d++) key[p * dim + d] += r[d];
      }
      const cnt = b - a || 1;
      for (let d = 0; d < dim; d++) key[p * dim + d] /= cnt;
    });
    let norm = 0;
    for (let i = 0; i < key.length; i++) norm += key[i] * key[i];
    norm = Math.sqrt(norm) + 1e-6;
    for (let i = 0; i < key.length; i++) key[i] /= norm;
    return key;
  }

  /**
   * [T][66] features + [T] speaking → {motion: Float32Array [T*C], ids}
   * Intent conditioning (RetrievalIndex.query): `targetArousal` (0..1, from the text)
   * adds arousal_bonus·(1 − |window_arousal − target|) to every window's score;
   * `intentTag === 'thinking'` adds thinking_bonus·max(0, still_then_move).
   * Both need the v2 index fields and are no-ops without them.
   */
  query(featRows, speaking, { targetArousal = null, intentTag = null } = {}) {
    const T = featRows.length;
    const C = this.channels.length;
    const win = this.win, hop = this.hop;
    const out = new Float32Array((T + win) * C);
    if (this.n === 0 || T === 0) return { motion: out.slice(0, T * C), ids: [] };
    const useArousal = this.arousal !== null && this.arousalBonus > 0 && targetArousal !== null && targetArousal !== undefined;
    const useThinking = intentTag === 'thinking' && this.stillThenMove !== null && this.thinkingBonus > 0;
    // edge-pad features and speaking by `win`
    const pad = featRows.slice();
    for (let i = 0; i < win; i++) pad.push(featRows[T - 1]);
    const spad = new Float32Array(T + win);
    for (let i = 0; i < T + win; i++) spad[i] = speaking[Math.min(i, T - 1)] ? 1 : 0;
    const cf = Math.min(this.crossfade, win);
    const w = [];
    for (let i = 1; i <= this.crossfade; i++) w.push(i / (this.crossfade + 1)); // np.linspace(0,1,cf+2)[1:-1]
    let prev = -1;
    const ids = [];
    const sims = new Float32Array(this.n);
    for (let h = 0; h < T; h += hop) {
      const key = this.windowKey(pad, h);
      let sMean = 0;
      for (let k = 0; k < win; k++) sMean += spad[h + k];
      sMean /= win;
      let best = -Infinity, j = 0;
      for (let i = 0; i < this.n; i++) {
        let s = 0;
        const off = i * this.keyDim;
        for (let d = 0; d < this.keyDim; d++) s += this.keys[off + d] * key[d];
        if (prev >= 0 && this.nextId[prev] === i) s += this.continuityBonus;
        s += this.speakingBonus * (1.0 - Math.abs(this.speaking[i] - sMean));
        if (useArousal) s += this.arousalBonus * (1.0 - Math.abs(this.arousal[i] - targetArousal));
        if (useThinking) s += this.thinkingBonus * Math.max(0.0, this.stillThenMove[i]);
        sims[i] = s;
        if (s > best) { best = s; j = i; }
      }
      const m = this.motion.subarray(j * win * C, (j + 1) * win * C);
      if (h === 0 || this.crossfade <= 0) {
        out.set(m, h * C);
      } else {
        for (let k = 0; k < cf; k++) {
          const a = w[k];
          for (let c = 0; c < C; c++) out[(h + k) * C + c] = (1 - a) * out[(h + k) * C + c] + a * m[k * C + c];
        }
        out.set(m.subarray(cf * C), (h + cf) * C);
      }
      prev = j;
      ids.push(j);
    }
    return { motion: out.slice(0, T * C), ids };
  }
}

// ---------------------------------------------------------------------------
// speech-envelope heuristic — animacy/serve.py envelope_motion (no model needed)
// ---------------------------------------------------------------------------
/**
 * @param {Float32Array[]} featRows  normalised [T][66] (energy = col 64, Δenergy = col 65)
 * @returns {{frames: object[], speaking: Uint8Array}}
 */
export function envelopeMotion(featRows, seed = 0, rateHz = RATE_HZ) {
  const n = featRows.length;
  const rng = seededRng(seed);
  const energy = new Float64Array(n), denergy = new Float64Array(n);
  for (let i = 0; i < n; i++) { energy[i] = featRows[i][64]; denergy[i] = featRows[i][65]; }
  const speaking = new Uint8Array(n);
  for (let i = 0; i < n; i++) speaking[i] = energy[i] > -0.3 ? 1 : 0;
  const k9 = hanning(9);
  let ks = 0; for (const v of k9) ks += v; for (let i = 0; i < 9; i++) k9[i] /= ks;
  const env = convolveSame(energy, k9);
  const onset = convolveSame(denergy, k9).map((v) => Math.max(v, 0));
  const k7 = hanning(7).map((v) => v / 3.5);
  const onsetSm = convolveSame(onset, k7);
  const ph = [rng() * 6, rng() * 6, rng() * 6];
  const lean = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) {
    if (speaking[i + 1] > speaking[i]) {
      const m = Math.min(45, n - i);
      for (let k = 0; k < m; k++) lean[i + k] += 40 * Math.exp(-k / 15.0);
    }
  }
  const frames = [];
  for (let i = 0; i < n; i++) {
    const t = i / rateHz;
    const f = neutralFrame();
    f.t = t;
    f.face_valid = 1;
    f.speaking = speaking[i];
    f.head_yaw = 6 * Math.sin(2 * Math.PI * 0.11 * t + ph[0]) + 3 * Math.sin(2 * Math.PI * 0.23 * t + ph[1]);
    f.head_roll = 3 * Math.sin(2 * Math.PI * 0.07 * t + ph[2]);
    f.head_pitch = -4 * Math.min(Math.max(env[i], -1), 2) - 6 * onsetSm[i];
    const brow = Math.min(Math.max(0.35 * (env[i] - 0.2), 0), 1);
    f.brow_l = brow; f.brow_r = brow;
    f.head_x = lean[i];
    f.torso_lean_fwd = lean[i] / 8.0;
    f.mouth_open = Math.min(Math.max(0.5 * (env[i] + 0.5), 0), 1) * speaking[i];
    frames.push(f);
  }
  return { frames, speaking };
}
