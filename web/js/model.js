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
// Sampling mirrors infer.sample_codes: z = logits/T + w·bigram[prev]; softmax;
// categorical draw. The RNG is sfc32 (not numpy's PCG64), so a seeded run is
// reproducible in the browser but not bit-identical to Python; with T → 0 both
// reduce to argmax and tests/test_web_model_parity.py checks that path exactly.

import { float16ToFloat32, seededRng, smoothColumns, hanning, convolveSame } from './dsp.js';
import { N_FEATS, RATE_HZ } from './features.js';
import { neutralFrame, BOUNDS } from './canonical.js';

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

// ---------------------------------------------------------------------------
// code samplers, keyed by model.json `arch` (default "a2m")
// ---------------------------------------------------------------------------
// A sampler turns 15 Hz features into codes. Signature:
//   async (model, feat15: Float32Array [L*66], L, speaking15: BigInt64Array [L],
//          {causal, temperature, bigramWeight, seed}) → Int32Array codes [L]
//
// "a2m" (v1): one non-autoregressive pass gives logits for every step, then the
//   bigram-prior categorical draw of infer.sample_codes.
// "ar" (planned v2): the graph is stepped once per code with its own previous
//   code(s) as input. Its per-step ONNX signature is declared in model.json
//   (`ar.inputs` / `ar.outputs`); register it here as SAMPLERS.ar when it lands.
export const SAMPLERS = {
  async a2m(model, feat15, L, speaking15, { causal, temperature, bigramWeight, seed }) {
    const lg = await model.logits(feat15, L, speaking15, causal);
    return sampleCodes(lg, L, model.nCodes, model.bigram, { temperature, bigramWeight, seed });
  },
};

// ---------------------------------------------------------------------------
// the learned model
// ---------------------------------------------------------------------------
export class MotionModel {
  constructor(meta, a2m, decoder, bigram) {
    this.meta = meta;
    this.a2m = a2m;
    this.decoder = decoder;
    this.bigram = bigram;               // Float32Array [n*n] or null
    this.channels = meta.channels;
    this.nCodes = meta.n_codes;
    this.arch = meta.arch || 'a2m';
    this.sampling = meta.sampling || { temperature: 0.8, bigram_weight: 0.5 };
    this.smoothHz = (meta.smoothing && meta.smoothing.cutoff_hz) || 6.0;
    if (!SAMPLERS[this.arch]) throw new Error(`model.json arch '${this.arch}' has no sampler in web/js/model.js (known: ${Object.keys(SAMPLERS).join(', ')})`);
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
    const files = [meta.a2m.file, meta.vq_decoder.file, meta.bigram && meta.bigram.file].filter(Boolean);
    const sizes = [meta.a2m.bytes || 0, meta.vq_decoder.bytes || 0, 0];
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
    const a2m = await ort.InferenceSession.create(bufs[0], opts);
    const decoder = await ort.InferenceSession.create(bufs[1], opts);
    let bigram = null;
    if (bufs[2]) {
      const u16 = new Uint16Array(bufs[2].buffer, bufs[2].byteOffset, bufs[2].byteLength >> 1);
      bigram = float16ToFloat32(u16);
      if (bigram.length !== meta.n_codes * meta.n_codes) { console.warn('[model] bigram size mismatch, ignoring prior'); bigram = null; }
    }
    say('model ready', 1);
    return new MotionModel(meta, a2m, decoder, bigram);
  }

  /** a2m logits for 15 Hz features. @returns {Promise<Float32Array>} flat [L*n] */
  async logits(feat15, L, speaking15, causal) {
    const ort = await loadOrt();
    const out = await this.a2m.run({
      features: new ort.Tensor('float32', feat15, [1, L, N_FEATS]),
      speaking: new ort.Tensor('int64', speaking15, [1, L]),
      causal: new ort.Tensor('int64', BigInt64Array.of(causal ? 1n : 0n), [1]),
    });
    return out.logits.data;
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
  async generate(featRows, speaking, { causal = false, temperature = null, bigramWeight = null, seed = 0, smoothHz = undefined } = {}) {
    const T = featRows.length;
    const C = this.channels.length;
    const stats = this.meta.stats || {};
    if (T < FRAMES_PER_CODE) {
      const m = new Float32Array(T * C);
      for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) m[t * C + c] = (stats.mean && stats.mean[c]) || 0;
      return { motion: m, codes: new Int32Array(0) };
    }
    const L = Math.floor(T / FRAMES_PER_CODE);
    const f15 = poolPairs(featRows, N_FEATS);
    const s15 = poolFlag(speaking);
    const codes = await SAMPLERS[this.arch](this, f15, L, s15, {
      causal,
      temperature: temperature ?? this.sampling.temperature ?? 0.8,
      bigramWeight: bigramWeight ?? this.sampling.bigram_weight ?? 0.5,
      seed,
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
    return { motion: m.slice(0, T * C), codes };
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
   */
  query(featRows, speaking) {
    const T = featRows.length;
    const C = this.channels.length;
    const win = this.win, hop = this.hop;
    const out = new Float32Array((T + win) * C);
    if (this.n === 0 || T === 0) return { motion: out.slice(0, T * C), ids: [] };
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
