// Record mode: capture the live canonical channel stream (30 Hz) + microphone
// audio with a shared clock, straight from the browser, and hand back the same
// files `animacy capture` writes — no server involved.
//
//   <subject>_<slug>/motion.json   animacy.schema.HumanClip.to_web_json shape:
//                                  {schema, rate_hz, n, channels, data:{channel:[...]}}
//   <subject>_<slug>/audio.webm    MediaRecorder (opus), starts at motion t = 0
//   <subject>_<slug>/meta.json     source "webcam-browser", role, neutral, license, tool versions
//
// A guided session (`SessionRunner`) walks a contributor through the same
// prompts as scripts/record_me.py, each with a countdown, and bundles all takes
// into one zip (STORE-only writer below; no dependency).

import { CHANNELS, FLAGS } from './canonical.js';

export const ANIMACY_WEB_VERSION = '0.2';
export const RATE_HZ = 30;

// (slug, role, seconds, spoken prompt) — mirrors scripts/record_me.py PROMPTS
export const PROMPTS = [
  { slug: 'neutral', role: 'speaking', seconds: 8, text: 'First, look at the camera with a relaxed face for a few seconds. This is your neutral pose.' },
  { slug: 'explain_project', role: 'speaking', seconds: 90, text: 'Explain what you are building this week, like you would to a friend who is not technical.' },
  { slug: 'funny_story', role: 'speaking', seconds: 60, text: 'Tell a funny story that actually happened to you.' },
  { slug: 'excited', role: 'speaking', seconds: 45, text: 'Now tell me about something you are genuinely excited about. Let it show.' },
  { slug: 'frustrated', role: 'speaking', seconds: 45, text: 'Now something that annoyed you recently. Complain a little.' },
  { slug: 'sad', role: 'speaking', seconds: 40, text: 'Now something you find a bit sad, quietly.' },
  { slug: 'questions', role: 'speaking', seconds: 60, text: 'Answer out loud: what is your favourite food, where would you travel tomorrow, and what would you tell yourself five years ago?' },
  { slug: 'yes_no', role: 'speaking', seconds: 40, text: 'Say yes and no in as many different ways as you can. Agree, disagree, hesitate, be certain.' },
  { slug: 'listen_podcast_1', role: 'listening', seconds: 120, text: 'Listening time. Start a podcast or a video on your phone, and just listen and react naturally. Do not talk.' },
  { slug: 'listen_podcast_2', role: 'listening', seconds: 120, text: 'Keep listening. React as you normally would, nod, frown, laugh, look away when you think.' },
  { slug: 'listen_disagree', role: 'listening', seconds: 60, text: 'Find something you disagree with, and listen to it. Do not talk, just react.' },
  { slug: 'greetings', role: 'speaking', seconds: 40, text: 'Greet me a few different ways, then say goodbye a few different ways.' },
];
export const QUICK = new Set(['neutral', 'explain_project', 'listen_podcast_1']);

const slugify = (s) => String(s || 'me').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 32) || 'me';

// ---------------------------------------------------------------------------
// 30 Hz resampling of the irregular webcam frame stream
// ---------------------------------------------------------------------------
/**
 * @param {{t:number, channels:object}[]} samples  t in seconds from the audio start, ascending
 * @param {number} seconds  requested length (frames past the last sample hold it)
 * @returns {{n:number, data:Object<string, (number|null)[]>}}
 */
export function resampleTo30Hz(samples, seconds) {
  const n = Math.max(0, Math.floor(seconds * RATE_HZ) + 1);
  const data = {};
  for (const c of CHANNELS) data[c] = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = i / RATE_HZ;
    data.t[i] = +t.toFixed(4);
    if (!samples.length) { for (const c of CHANNELS) if (c !== 't') data[c][i] = FLAGS.includes(c) ? 0 : null; continue; }
    while (j < samples.length - 1 && samples[j + 1].t <= t) j++;
    const a = samples[j];
    const b = j < samples.length - 1 ? samples[j + 1] : null;
    const useB = b && t > a.t;
    const w = useB ? Math.min(1, (t - a.t) / Math.max(b.t - a.t, 1e-6)) : 0;
    const near = useB && w > 0.5 ? b : a;
    for (const c of CHANNELS) {
      if (c === 't') continue;
      if (FLAGS.includes(c)) { data[c][i] = near.channels[c] ? 1 : 0; continue; }
      const va = a.channels[c], vb = useB ? b.channels[c] : va;
      const ok = (v) => v !== null && v !== undefined && !Number.isNaN(v);
      let v;
      if (ok(va) && ok(vb)) v = va + (vb - va) * w;
      else if (ok(va) && w < 0.5) v = va;
      else if (ok(vb) && w >= 0.5) v = vb;
      else v = null;
      data[c][i] = v === null ? null : +v.toFixed(3);
    }
  }
  return { n, data };
}

// ---------------------------------------------------------------------------
// minimal zip writer (STORE) — good enough for a few MB of JSON + webm
// ---------------------------------------------------------------------------
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function dosDateTime(d = new Date()) {
  const time = ((d.getHours() & 31) << 11) | ((d.getMinutes() & 63) << 5) | ((d.getSeconds() >> 1) & 31);
  const date = (((d.getFullYear() - 1980) & 127) << 9) | (((d.getMonth() + 1) & 15) << 5) | (d.getDate() & 31);
  return { time, date };
}

/**
 * @param {{name:string, data:Uint8Array|string}[]} entries
 * @returns {Blob} application/zip
 */
export function makeZip(entries) {
  const enc = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;
  const { time, date } = dosDateTime();
  for (const e of entries) {
    const name = enc.encode(e.name);
    const data = typeof e.data === 'string' ? enc.encode(e.data) : e.data;
    const crc = crc32(data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);       // version needed
    local.setUint16(6, 0x0800, true);   // UTF-8 names
    local.setUint16(8, 0, true);        // STORE
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    parts.push(new Uint8Array(local.buffer), name, data);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0x0800, true);
    cd.setUint16(10, 0, true);
    cd.setUint16(12, time, true);
    cd.setUint16(14, date, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, data.length, true);
    cd.setUint32(24, data.length, true);
    cd.setUint16(28, name.length, true);
    cd.setUint16(30, 0, true);
    cd.setUint16(32, 0, true);
    cd.setUint16(34, 0, true);
    cd.setUint16(36, 0, true);
    cd.setUint32(38, 0, true);
    cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), name);
    offset += 30 + name.length + data.length;
  }
  let cdSize = 0;
  for (const c of central) cdSize += c.length;
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(4, 0, true);
  end.setUint16(6, 0, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, offset, true);
  end.setUint16(20, 0, true);
  return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: 'application/zip' });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 2000);
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------
export class Recorder {
  /**
   * @param {object} o
   * @param {import('./motion_source.js').WebcamSource} o.webcam  running webcam source (frames + calibrator + optional mic)
   * @param {(msg:string)=>void} [o.onStatus]
   */
  constructor({ webcam, onStatus = null }) {
    this.webcam = webcam;
    this.onStatus = onStatus || (() => {});
    this.recording = false;
    this.samples = [];
    this.t0 = null;                 // performance.now()/1000 when the audio actually started
    this.rec = null;
    this.chunks = [];
    this.take = null;               // last finished take
    this.takes = [];
    this.mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
    this._stream = null;
    this._ownStream = false;
    this.meta = null;
  }

  get seconds() { return this.recording && this.t0 !== null ? performance.now() / 1000 - this.t0 : (this.take ? this.take.seconds : 0); }

  /** Called with every webcam frame (main.js forwards WebcamSource frames here). */
  onFrame(frame) {
    if (!this.recording || this.t0 === null || !frame || !frame.channels) return;
    const t = frame.t - this.t0;
    if (t < 0) return;
    this.samples.push({ t, channels: frame.channels });
  }

  async _audioStream() {
    const mic = this.webcam && this.webcam._audio && this.webcam._audio.mic;
    if (mic && mic.getAudioTracks().length) { this._ownStream = false; return mic; }
    const s = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    this._ownStream = true;
    return s;
  }

  /**
   * @param {object} o
   * @param {string} o.subject
   * @param {string} o.slug
   * @param {'speaking'|'listening'} o.role
   * @param {string} [o.prompt]
   * @param {number} [o.seconds]  planned length (for meta only)
   */
  async start({ subject = 'me', slug = 'take', role = 'speaking', prompt = '', seconds = 0 } = {}) {
    if (this.recording) throw new Error('already recording');
    if (!this.webcam || !this.webcam.running) throw new Error('start Webcam live first');
    this.samples = [];
    this.chunks = [];
    this.take = null;
    this.meta = { subject: slugify(subject), slug: slugify(slug), role, prompt, planned_seconds: seconds };
    this._stream = await this._audioStream();
    if (!window.MediaRecorder) throw new Error('MediaRecorder is not available in this browser');
    this.rec = new MediaRecorder(this._stream, this.mime ? { mimeType: this.mime } : undefined);
    this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    const started = new Promise((resolve) => { this.rec.onstart = () => resolve(); });
    this.rec.start(250);
    await started;
    this.t0 = performance.now() / 1000;   // motion t = 0 is the first audio sample
    this.recording = true;
    this.onStatus(`recording ${this.meta.subject}_${this.meta.slug} (${role})`);
  }

  /** @returns {Promise<object>} the take: {name, seconds, n, motion, meta, audio: Blob, zip: Blob} */
  async stop() {
    if (!this.recording) throw new Error('not recording');
    const seconds = performance.now() / 1000 - this.t0;
    const stopped = new Promise((resolve) => { this.rec.onstop = () => resolve(); });
    this.rec.stop();
    await stopped;
    this.recording = false;
    if (this._ownStream && this._stream) for (const tr of this._stream.getTracks()) tr.stop();
    this._stream = null;
    const { n, data } = resampleTo30Hz(this.samples, seconds);
    if (this.meta.role === 'listening') data.speaking = data.speaking.map(() => 0); // the mic hears the podcast, not you
    const motion = { schema: 'animacy.human.v1', rate_hz: RATE_HZ, n, channels: CHANNELS.slice(), data };
    const audio = new Blob(this.chunks, { type: this.mime || 'audio/webm' });
    const wc = this.webcam;
    const meta = {
      schema: 'animacy.human.v1',
      source: 'webcam-browser',
      rate_hz: RATE_HZ,
      subject: this.meta.subject,
      role: this.meta.role,
      prompt: { slug: this.meta.slug, text: this.meta.prompt, planned_seconds: this.meta.planned_seconds },
      arm: wc.arm,
      neutral: (wc.calibrator && wc.calibrator.neutral) || null,
      license: 'CC-BY-4.0',
      audio: { file: 'audio.webm', mime: audio.type, offset_s: 0.0, bytes: audio.size },
      seconds: +seconds.toFixed(3),
      n_frames: n,
      frames_captured: this.samples.length,
      face_valid_fraction: n ? +(data.face_valid.reduce((a, b) => a + b, 0) / n).toFixed(3) : 0,
      created: new Date().toISOString(),
      tool_versions: {
        animacy_web: ANIMACY_WEB_VERSION,
        mediapipe_tasks_vision: '0.10.21',
        face_model: 'face_landmarker/float16/1',
        pose_model: 'pose_landmarker_lite/float16/1',
        browser: navigator.userAgent,
      },
    };
    const name = `${this.meta.subject}_${this.meta.slug}`;
    const audioBytes = new Uint8Array(await audio.arrayBuffer());
    const zip = makeZip([
      { name: `${name}/motion.json`, data: JSON.stringify(motion) },
      { name: `${name}/audio.webm`, data: audioBytes },
      { name: `${name}/meta.json`, data: JSON.stringify(meta, null, 1) },
    ]);
    this.take = { name, seconds, n, motion, meta, audio, audioBytes, zip };
    this.takes.push(this.take);
    this.onStatus(`saved ${name}: ${n} frames, ${seconds.toFixed(1)} s, audio ${(audio.size / 1024).toFixed(0)} kB`);
    return this.take;
  }

  download(take = this.take) {
    if (!take) return;
    downloadBlob(take.zip, `${take.name}.zip`);
  }

  /** One zip with every take of this page session. */
  downloadSession(subject = null) {
    if (!this.takes.length) return null;
    const entries = [];
    for (const t of this.takes) {
      entries.push({ name: `${t.name}/motion.json`, data: JSON.stringify(t.motion) });
      entries.push({ name: `${t.name}/audio.webm`, data: t.audioBytes });
      entries.push({ name: `${t.name}/meta.json`, data: JSON.stringify(t.meta, null, 1) });
    }
    const zip = makeZip(entries);
    downloadBlob(zip, `animacy_session_${slugify(subject || this.takes[0].meta.subject)}_${new Date().toISOString().slice(0, 10)}.zip`);
    return zip;
  }
}

// ---------------------------------------------------------------------------
// Guided session
// ---------------------------------------------------------------------------
export class SessionRunner {
  /**
   * @param {object} o
   * @param {Recorder} o.recorder
   * @param {string} o.subject
   * @param {boolean} [o.quick]
   * @param {(state:object)=>void} o.onState   UI callback: {phase, prompt, index, total, remaining, done}
   * @param {boolean} [o.speak]  read prompts aloud with the Web Speech API
   */
  constructor({ recorder, subject, quick = false, onState, speak = true }) {
    this.recorder = recorder;
    this.subject = subject;
    this.prompts = quick ? PROMPTS.filter((p) => QUICK.has(p.slug)) : PROMPTS.slice();
    this.onState = onState || (() => {});
    this.speak = speak;
    this.index = -1;
    this.abort = false;
    this.skip = false;
    this.running = false;
    this.done = [];
  }

  get totalSeconds() { return this.prompts.reduce((a, p) => a + p.seconds, 0); }

  _say(text) {
    if (!this.speak || !window.speechSynthesis) return;
    try { window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(text)); } catch (e) { /* ignore */ }
  }

  _wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async run() {
    this.running = true;
    this.abort = false;
    this._say('Recording session. Sit facing the camera. Each prompt is read before its segment starts.');
    for (let i = 0; i < this.prompts.length && !this.abort; i++) {
      const p = this.prompts[i];
      this.index = i;
      this.skip = false;
      this.onState({ phase: 'prompt', prompt: p, index: i, total: this.prompts.length, remaining: 0, done: this.done });
      this._say(p.text);
      // read time + countdown
      const readMs = Math.min(9000, 1500 + 60 * p.text.length);
      for (let ms = readMs; ms > 0 && !this.abort && !this.skip; ms -= 250) {
        this.onState({ phase: 'countdown', prompt: p, index: i, total: this.prompts.length, remaining: Math.ceil(ms / 1000), done: this.done });
        await this._wait(250);
      }
      if (this.abort) break;
      if (this.skip) continue;
      this._say('three, two, one');
      for (let s = 3; s > 0 && !this.abort && !this.skip; s--) {
        this.onState({ phase: 'countdown', prompt: p, index: i, total: this.prompts.length, remaining: s, done: this.done });
        await this._wait(1000);
      }
      if (this.abort) break;
      if (this.skip) continue;
      try {
        await this.recorder.start({ subject: this.subject, slug: p.slug, role: p.role, prompt: p.text, seconds: p.seconds });
      } catch (e) {
        this.onState({ phase: 'error', prompt: p, index: i, total: this.prompts.length, remaining: 0, done: this.done, error: e.message });
        break;
      }
      const t0 = performance.now();
      while (performance.now() - t0 < p.seconds * 1000 && !this.abort && !this.skip) {
        this.onState({ phase: 'recording', prompt: p, index: i, total: this.prompts.length, remaining: Math.ceil(p.seconds - (performance.now() - t0) / 1000), done: this.done });
        await this._wait(250);
      }
      const take = await this.recorder.stop();
      this.done.push(take);
      this._say('Got it.');
      this.onState({ phase: 'saved', prompt: p, index: i, total: this.prompts.length, remaining: 0, done: this.done, take });
      await this._wait(1200);
    }
    this.running = false;
    this.onState({ phase: this.abort ? 'aborted' : 'complete', prompt: null, index: this.index, total: this.prompts.length, remaining: 0, done: this.done });
    if (!this.abort) this._say('Session complete. Thank you.');
    return this.done;
  }
}
