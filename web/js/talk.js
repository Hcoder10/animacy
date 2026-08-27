// Talk mode: text → Kokoro TTS (in the browser) → audio features → motion
// (model / retrieval / envelope) → canonical frames → both robots, while the
// waveform plays through WebAudio. Motion is clocked to `AudioContext.currentTime`,
// so speech and movement cannot drift apart.
//
// Listen mode (hook): microphone → energy VAD → causal model with speaking=0.
// Wired as `ListenSource` below; see its docstring for what is and is not done.

import { MotionSource } from './motion_source.js';
import { Track } from './clips.js';
import { CHANNELS } from './canonical.js';
import { audioFeatures, SR, RATE_HZ } from './features.js';
import { MotionModel, RetrievalIndex, envelopeMotion, motionToFrames } from './model.js';

export const KOKORO_VERSION = '1.2.1';
export const KOKORO_URL = `https://cdn.jsdelivr.net/npm/kokoro-js@${KOKORO_VERSION}/dist/kokoro.web.js`;
export const KOKORO_MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';
export const KOKORO_VOICES = ['af_heart', 'af_bella', 'af_nicole', 'am_adam', 'am_michael', 'bf_emma', 'bm_george'];

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------
let _ttsPromise = null;

/** Kokoro-82M (q8) in the browser via kokoro-js; WebGPU when available, wasm otherwise. Cached by the browser. */
export function loadTts(onProgress = null) {
  if (_ttsPromise) return _ttsPromise;
  _ttsPromise = (async () => {
    const say = (m, f) => { if (onProgress) onProgress(m, f); };
    say('loading kokoro-js…', 0.01);
    const { KokoroTTS } = await import(KOKORO_URL);
    const files = new Map();
    const progress = (p) => {
      if (p.status === 'progress' && p.file) {
        files.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
        let l = 0, t = 0;
        for (const v of files.values()) { l += v.loaded; t += v.total; }
        say(`downloading Kokoro TTS ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB (cached after the first time)`, t ? 0.02 + 0.9 * (l / t) : 0.02);
      } else if (p.status === 'ready') say('TTS ready', 1);
    };
    const wantGpu = !!navigator.gpu;
    try {
      const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL, { dtype: 'q8', device: wantGpu ? 'webgpu' : 'wasm', progress_callback: progress });
      tts._device = wantGpu ? 'webgpu' : 'wasm';
      say(`TTS ready (${tts._device})`, 1);
      return tts;
    } catch (e) {
      if (!wantGpu) throw e;
      console.warn('[talk] Kokoro on WebGPU failed, retrying on wasm:', e && e.message);
      const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL, { dtype: 'q8', device: 'wasm', progress_callback: progress });
      tts._device = 'wasm';
      say('TTS ready (wasm)', 1);
      return tts;
    }
  })();
  _ttsPromise.catch(() => { _ttsPromise = null; });
  return _ttsPromise;
}

/** Any sample rate → 16 kHz mono Float32Array via OfflineAudioContext (browser-native resampler). */
export async function resampleTo16k(audio, sr) {
  if (sr === SR) return Float32Array.from(audio);
  const n = Math.ceil((audio.length * SR) / sr);
  const ctx = new OfflineAudioContext(1, n, SR);
  const buf = ctx.createBuffer(1, audio.length, sr);
  buf.copyToChannel(Float32Array.from(audio), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  return out.getChannelData(0).slice();
}

// ---------------------------------------------------------------------------
// motion backends
// ---------------------------------------------------------------------------
/**
 * Owns the (lazily loaded) motion model and retrieval index. `backend` is one of
 * 'model' | 'retrieval' | 'envelope'; unavailable backends fall back to the
 * envelope heuristic and say so.
 */
export class MotionBackends {
  constructor({ baseUrl = 'models/', bundle = null, onStatus = null } = {}) {
    this.baseUrl = baseUrl;
    this.bundle = bundle || {};   // from manifest.json: {model_json, a2m, vq_decoder, bigram, retrieval}
    this.onStatus = onStatus || (() => {});
    this.model = null;
    this.index = null;
    this._modelPromise = null;
    this._indexPromise = null;
  }

  get hasModel() { return !!((this.bundle.a2m || this.bundle.a2m_ar) && this.bundle.vq_decoder && this.bundle.model_json); }
  get hasRetrieval() { return !!this.bundle.retrieval; }

  /** Backends in picker order: the bundle's `default_backend` first (retrieval unless the model earned it). */
  available() {
    const out = ['envelope'];
    if (this.hasModel) out.unshift('model');
    if (this.hasRetrieval) out.unshift('retrieval');
    const def = this.bundle.default_backend;
    if (def && out.includes(def)) { out.splice(out.indexOf(def), 1); out.unshift(def); }
    return out;
  }

  async getModel() {
    if (!this.hasModel) return null;
    if (!this._modelPromise) {
      this._modelPromise = MotionModel.load(this.baseUrl, (m, f) => this.onStatus(m, f)).then((m) => { this.model = m; return m; });
      this._modelPromise.catch(() => { this._modelPromise = null; });
    }
    return this._modelPromise;
  }

  async getIndex() {
    if (!this.hasRetrieval) return null;
    if (!this._indexPromise) {
      this._indexPromise = RetrievalIndex.load(this.baseUrl, (m, f) => this.onStatus(m, f)).then((i) => { this.index = i; return i; });
      this._indexPromise.catch(() => { this._indexPromise = null; });
    }
    return this._indexPromise;
  }

  /**
   * features [T][66] + speaking [T] → {frames, backend, codes?}
   * @param {string} backend  requested backend; the one actually used is returned
   */
  async motion(featRows, speaking, backend, { causal = false, seed = 0 } = {}) {
    const T = featRows.length;
    if (backend === 'model') {
      try {
        const model = await this.getModel();
        if (model) {
          const onStep = model.arch === 'ar' ? (i, L) => this.onStatus(`model (${model.describe}) step ${i}/${L}…`, i / L) : null;
          const { motion, codes, arch } = await model.generate(featRows, speaking, { causal, seed, onStep });
          return { frames: motionToFrames(motion, T, model.channels, speaking), backend: 'model', codes, arch };
        }
      } catch (e) {
        console.warn('[talk] model backend failed, falling back:', e && e.message);
        this.onStatus(`model failed (${e && e.message}); using ${this.hasRetrieval ? 'retrieval' : 'envelope'}`);
      }
      backend = this.hasRetrieval ? 'retrieval' : 'envelope';
    }
    if (backend === 'retrieval') {
      try {
        const index = await this.getIndex();
        if (index) {
          const { motion, ids } = index.query(featRows, speaking);
          return { frames: motionToFrames(motion, T, index.channels, speaking), backend: 'retrieval', ids };
        }
      } catch (e) {
        console.warn('[talk] retrieval backend failed, falling back:', e && e.message);
        this.onStatus(`retrieval failed (${e && e.message}); using envelope`);
      }
      backend = 'envelope';
    }
    const { frames } = envelopeMotion(featRows, seed);
    return { frames, backend: 'envelope' };
  }
}

function framesToTrack(frames, name, rateHz = RATE_HZ) {
  const n = frames.length;
  const t = Array.from({ length: n }, (_, i) => i / rateHz);
  const data = {};
  for (const c of CHANNELS) if (c !== 't') data[c] = frames.map((f) => f[c]);
  return new Track({ kind: 'canonical', name, t, data, group: 'talk' });
}

// ---------------------------------------------------------------------------
// TalkSource: transport-compatible, clocked to the audio
// ---------------------------------------------------------------------------
export class TalkSource extends MotionSource {
  /**
   * @param {object} o
   * @param {MotionBackends} o.backends
   * @param {string} [o.backend]  'model' | 'retrieval' | 'envelope'
   * @param {(msg:string, frac?:number)=>void} [o.onStatus]
   */
  constructor({ backends, backend = 'model', onStatus = null, loop = false }) {
    super();
    this.backends = backends;
    this.backend = backend;
    this.onStatus = onStatus || (() => {});
    this.ctx = null;
    this.buffer = null;         // AudioBuffer of the last utterance
    this.node = null;           // playing AudioBufferSourceNode
    this.track = null;
    this.loop = loop;
    this.speed = 1.0;           // fixed: the audio sets the pace
    this.playing = false;
    this.finished = false;
    this._offset = 0;           // clip time at the last start/pause
    this._startedAt = 0;        // ctx.currentTime when playback (re)started
    this.busy = false;
    this.last = null;           // {text, backend, seconds, ms}
  }

  get duration() { return this.track ? this.track.duration : 0; }
  get time() {
    if (!this.track) return 0;
    if (!this.playing) return this._offset;
    return Math.min(this.duration, this._offset + (this.ctx.currentTime - this._startedAt));
  }

  _ensureCtx() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  }

  /** The audio clock only runs once the context is; wait for it (a user gesture normally makes this instant). */
  async _ensureRunning() {
    const ctx = this._ensureCtx();
    if (ctx.state !== 'running') {
      try { await Promise.race([ctx.resume(), new Promise((r) => setTimeout(r, 1500))]); } catch (e) { /* stays suspended */ }
    }
    if (ctx.state !== 'running') this.onStatus('audio is blocked until you click somewhere on the page');
    return ctx;
  }

  /** Synthesise + animate `text`. Resolves when playback has started. */
  async say(text, { voice = 'af_heart', seed = 0 } = {}) {
    if (this.busy) throw new Error('still working on the previous line');
    text = (text || '').trim();
    if (!text) throw new Error('nothing to say');
    this.busy = true;
    try {
      const t0 = performance.now();
      const tts = await loadTts((m, f) => this.onStatus(m, f));
      this.onStatus(`synthesising (${tts._device})…`, 0);
      const raw = await tts.generate(text, { voice });
      const tTts = performance.now() - t0;
      return await this.sayAudio(raw.audio, raw.sampling_rate, { seed, text, ttsMs: tTts });
    } finally {
      this.busy = false;
    }
  }

  /**
   * Animate an existing waveform (any sample rate). Used by `say` and by the
   * verification suite, which injects a synthetic voice instead of running TTS.
   */
  async sayAudio(audio, sr, { seed = 0, text = '(audio)', ttsMs = 0 } = {}) {
    const ctx = await this._ensureRunning();
    this.onStatus('features…');
    const wav16 = await resampleTo16k(audio, sr);
    const nTicks = Math.ceil((wav16.length / SR) * RATE_HZ);
    const feats = audioFeatures(wav16, nTicks);
    // talk mode (serve._speaking_from_audio): the robot is the speaker wherever its own voice has energy
    const speaking = new Uint8Array(nTicks);
    for (let i = 0; i < nTicks; i++) speaking[i] = feats[i][64] > -0.3 ? 1 : 0;
    this.onStatus(`motion (${this.backend})…`);
    const t1 = performance.now();
    const res = await this.backends.motion(feats, speaking, this.backend, { causal: false, seed });
    const tMotion = performance.now() - t1;
    this.track = framesToTrack(res.frames, text.slice(0, 40));
    this.buffer = ctx.createBuffer(1, audio.length, sr);
    this.buffer.copyToChannel(Float32Array.from(audio), 0);
    this.last = { text, backend: res.backend, arch: res.arch || null, seconds: audio.length / sr, ttsMs, motionMs: tMotion, codes: res.codes ? res.codes.length : 0, frames: res.frames.length };
    this._offset = 0;
    this.finished = false;
    this._startAudio(0);
    const label = res.backend === 'model' && res.arch ? `model/${res.arch}` : res.backend;
    this.onStatus(`${label}: ${(audio.length / sr).toFixed(1)} s of speech · tts ${(ttsMs / 1000).toFixed(1)} s · motion ${tMotion.toFixed(0)} ms`, 1);
    return this.last;
  }

  _stopAudio() {
    if (this.node) { try { this.node.onended = null; this.node.stop(); } catch (e) { /* already stopped */ } this.node = null; }
  }

  _startAudio(offset) {
    const ctx = this._ensureCtx();
    this._stopAudio();
    if (!this.buffer) return;
    const node = ctx.createBufferSource();
    node.buffer = this.buffer;
    node.connect(ctx.destination);
    node.onended = () => {
      if (this.node !== node) return;
      this.node = null;
      if (this.loop) { this._offset = 0; this._startAudio(0); }
      else { this._offset = this.duration; this.playing = false; this.finished = true; }
    };
    node.start(0, Math.min(offset, Math.max(0, this.buffer.duration - 0.01)));
    this.node = node;
    this._offset = offset;
    this._startedAt = ctx.currentTime;
    this.playing = true;
    this.finished = false;
  }

  play() { if (!this.track) return; if (this.finished) this._offset = 0; if (!this.playing) this._ensureRunning().then(() => { if (!this.playing) this._startAudio(this._offset); }); }
  pause() { if (!this.playing) return; this._offset = this.time; this._stopAudio(); this.playing = false; }
  toggle() { this.playing ? this.pause() : this.play(); }
  seek(t) {
    if (!this.track) return;
    const was = this.playing;
    this._stopAudio();
    this._offset = Math.min(Math.max(t, 0), this.duration);
    this.playing = false;
    this.finished = false;
    if (was) this._startAudio(this._offset);
    this._pendingSeek = true;
  }

  update(_realDt) {
    if (!this.track) return null;
    const t = this.time;
    const seek = !!this._pendingSeek;
    const dt = Math.max(1e-4, this.playing ? t - (this._lastT ?? t) : 1 / RATE_HZ);
    this._pendingSeek = false;
    this._lastT = t;
    const frame = { t, dt, seek, channels: this.track.sample(t) };
    this._emit(frame);
    return frame;
  }

  stop() {
    this._stopAudio();
    this.playing = false;
  }
}

// ---------------------------------------------------------------------------
// ListenSource (hook): microphone → VAD → causal model, speaking = 0
// ---------------------------------------------------------------------------
/**
 * What is wired: mic capture at 16 kHz (AudioWorklet-free: ScriptProcessor for
 * portability), a 2 s rolling window, an energy VAD, and — every `hopS` while the
 * person is talking — a causal (`speaking = 0`) pass of the selected backend over
 * the window whose last `hopS` of frames are appended to a live queue that the
 * viewer drains at 30 Hz. Per-utterance feature normalisation is approximated by
 * normalising the rolling window.
 * What is NOT wired: the gaze overlay from the webcam face position (the
 * `gaze` field is reserved) and any tuning on real conversations.
 */
export class ListenSource extends MotionSource {
  constructor({ backends, backend = 'model', onStatus = null, windowS = 2.0, hopS = 0.5 }) {
    super();
    this.backends = backends;
    this.backend = backend;
    this.onStatus = onStatus || (() => {});
    this.windowS = windowS;
    this.hopS = hopS;
    this.ctx = null;
    this.stream = null;
    this.proc = null;
    this.ring = new Float32Array(Math.round(SR * windowS));
    this.ringFill = 0;
    this.sinceHop = 0;
    this.level = 0;
    this.noise = 0.01;
    this.talking = false;
    this.queue = [];            // canonical frames waiting to be played
    this.running = false;
    this._acc = 0;
    this._busy = false;
    this.gaze = null;           // reserved: {yaw, pitch} from the webcam face position
  }

  async start() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.proc = this.ctx.createScriptProcessor(1024, 1, 1);
    this.proc.onaudioprocess = (e) => this._onAudio(e.inputBuffer.getChannelData(0));
    src.connect(this.proc);
    this.proc.connect(this.ctx.destination);
    this.running = true;
    this.onStatus('listening — talk to the robots');
  }

  _onAudio(chunk) {
    const n = chunk.length;
    // rolling ring: shift left, append
    this.ring.copyWithin(0, n);
    this.ring.set(chunk, this.ring.length - n);
    this.ringFill = Math.min(this.ring.length, this.ringFill + n);
    let s = 0;
    for (let i = 0; i < n; i++) s += chunk[i] * chunk[i];
    const rms = Math.sqrt(s / n);
    this.noise = Math.min(this.noise * 1.002 + 1e-5, Math.max(rms, 1e-4));
    this.level = 0.8 * this.level + 0.2 * rms;
    this.talking = this.level > Math.max(0.015, this.noise * 3.5);
    this.sinceHop += n / SR;
    if (this.talking && this.sinceHop >= this.hopS && this.ringFill >= this.ring.length && !this._busy) {
      this.sinceHop = 0;
      this._infer();
    }
  }

  async _infer() {
    this._busy = true;
    try {
      const feats = audioFeatures(this.ring.slice());
      const speaking = new Uint8Array(feats.length);      // 0: the robot is listening
      const res = await this.backends.motion(feats, speaking, this.backend, { causal: true, seed: (Date.now() / 1000) | 0 });
      const keep = Math.round(this.hopS * RATE_HZ);
      const tail = res.frames.slice(-keep);
      if (this.queue.length > 2 * keep) this.queue.length = keep; // do not fall behind
      this.queue.push(...tail);
    } catch (e) {
      console.warn('[listen] inference failed:', e && e.message);
    } finally {
      this._busy = false;
    }
  }

  update(realDt) {
    if (!this.running) return null;
    this._acc += realDt;
    if (this._acc < 1 / RATE_HZ) return null;
    this._acc = 0;
    const f = this.queue.length ? this.queue.shift() : null;
    if (!f) return null;
    return { t: performance.now() / 1000, dt: 1 / RATE_HZ, channels: f };
  }

  stop() {
    this.running = false;
    try { if (this.proc) { this.proc.disconnect(); this.proc.onaudioprocess = null; } } catch (e) { /* ignore */ }
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    if (this.ctx) { try { this.ctx.close(); } catch (e) { /* ignore */ } }
    this.ctx = this.stream = this.proc = null;
    this.onStatus('stopped');
  }
}
