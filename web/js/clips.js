// Clip parsing + sampling for the viewer.
//
// Three on-disk shapes, one in-memory shape (`Track`):
//   * Autonomous OS CSV   (robots/lamp/clips/native/*.csv)
//       timestamp,<joint>.pos,...   20 Hz, seconds      → kind 'joint'
//   * joint-table JSON    (animacy/export.py `json` format, Reachy native clips)
//       {"robot","rate_hz","joints","t","data":{joint:[...]}}   → kind 'joint'
//   * canonical clip JSON (animacy/schema.py `HumanClip.to_web_json`)
//       {"schema":"animacy.human.v1","rate_hz","n","channels","data":{ch:[...]}}
//                                                                → kind 'canonical'
// A Track samples any column at an arbitrary time by linear interpolation
// (hold at both ends), so 20 Hz vendor data and 30 Hz canonical data both play
// smoothly at display rate.

import { CHANNELS, neutralFrame } from './canonical.js';

export class Track {
  /**
   * @param {object} o
   * @param {'joint'|'canonical'} o.kind
   * @param {string} o.name   display name
   * @param {string} [o.robot]  for 'joint' tracks: the profile name the columns belong to
   * @param {number[]} o.t     strictly increasing seconds, t[0] === 0
   * @param {Object<string, number[]>} o.data  column → values (null/NaN allowed)
   * @param {string} [o.group]  UI grouping label
   */
  constructor({ kind, name, robot = null, t, data, group = '' }) {
    this.kind = kind;
    this.name = name;
    this.robot = robot;
    this.group = group;
    this.t = t;
    this.data = data;
    this.columns = Object.keys(data);
    this.duration = t.length ? t[t.length - 1] : 0;
    this._cursor = 0;
  }

  get n() { return this.t.length; }

  _locate(time) {
    const t = this.t;
    const n = t.length;
    if (n === 0) return [0, 0, 0];
    if (time <= t[0]) return [0, 0, 0];
    if (time >= t[n - 1]) return [n - 1, n - 1, 0];
    let i = this._cursor;
    if (i >= n - 1 || t[i] > time) i = 0;
    while (i < n - 2 && t[i + 1] <= time) i++;
    this._cursor = i;
    const span = t[i + 1] - t[i];
    const a = span > 0 ? (time - t[i]) / span : 0;
    return [i, i + 1, a];
  }

  /** Sample every column at `time` (seconds). Missing values → 0 for canonical, hold for joints. */
  sample(time) {
    const [i, j, a] = this._locate(time);
    const out = {};
    for (const c of this.columns) {
      const col = this.data[c];
      let v0 = col[i], v1 = col[j];
      if (v0 === null || v0 === undefined || Number.isNaN(v0)) v0 = v1;
      if (v1 === null || v1 === undefined || Number.isNaN(v1)) v1 = v0;
      if (v0 === null || v0 === undefined || Number.isNaN(v0)) { out[c] = this.kind === 'canonical' ? 0 : NaN; continue; }
      out[c] = v0 + (v1 - v0) * a;
    }
    if (this.kind === 'canonical') {
      out.t = time;
      // flags are not interpolated: nearest
      for (const f of ['arm_valid', 'speaking', 'face_valid']) if (f in out) out[f] = out[f] >= 0.5 ? 1 : 0;
    }
    return out;
  }
}

/** Autonomous OS `hal/recordings/*.csv` → joint Track (strips `.pos`, t relative). */
export function parseAutonomousCsv(text, name, robot = 'lamp') {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.trim().length);
  if (lines.length < 2) throw new Error(`${name}: empty csv`);
  const header = lines[0].split(',').map((s) => s.trim());
  const ti = header.indexOf('timestamp');
  if (ti < 0) throw new Error(`${name}: no timestamp column`);
  const cols = header.map((h) => (h.endsWith('.pos') ? h.slice(0, -4) : h));
  const t = [];
  const data = {};
  for (let k = 0; k < header.length; k++) if (k !== ti) data[cols[k]] = [];
  let t0 = null;
  for (let r = 1; r < lines.length; r++) {
    const parts = lines[r].split(',');
    if (parts.length < header.length) continue;
    const ts = parseFloat(parts[ti]);
    if (Number.isNaN(ts)) continue;
    if (t0 === null) t0 = ts;
    const tt = ts - t0;
    if (t.length && tt <= t[t.length - 1]) continue; // keep strictly increasing
    t.push(tt);
    for (let k = 0; k < header.length; k++) {
      if (k === ti) continue;
      const v = parseFloat(parts[k]);
      data[cols[k]].push(Number.isNaN(v) ? null : v);
    }
  }
  return new Track({ kind: 'joint', name, robot, t, data, group: 'native' });
}

/** animacy `json` joint table → joint Track. */
export function parseJointJson(obj, name, robot = null) {
  const joints = obj.joints || Object.keys(obj.data || {});
  const data = {};
  for (const j of joints) data[j] = obj.data[j];
  let t = obj.t;
  if (!t) {
    const n = data[joints[0]] ? data[joints[0]].length : 0;
    const rate = obj.rate_hz || 30;
    t = Array.from({ length: n }, (_, i) => i / rate);
  }
  if (t.length && t[0] !== 0) t = t.map((x) => x - t[0]);
  return new Track({ kind: 'joint', name, robot: robot || obj.robot || null, t, data, group: 'native' });
}

/** `HumanClip.to_web_json` output → canonical Track. */
export function parseCanonicalJson(obj, name) {
  const channels = obj.channels || Object.keys(obj.data || {});
  const data = {};
  for (const c of channels) if (obj.data[c]) data[c] = obj.data[c];
  let n = obj.n;
  if (n === undefined) n = data[channels[0]] ? data[channels[0]].length : 0;
  const rate = obj.rate_hz || 30;
  let t;
  if (data.t && data.t.length === n && data.t.every((v) => v !== null && !Number.isNaN(v))) t = data.t.slice();
  else t = Array.from({ length: n }, (_, i) => i / rate);
  if (t.length && t[0] !== 0) t = t.map((x) => x - t[0]);
  delete data.t;
  return new Track({ kind: 'canonical', name, t, data, group: obj.group || 'captured' });
}

// ---------------------------------------------------------------------------
// Built-in synthetic calibration clips
// ---------------------------------------------------------------------------
// These exist so the sign conventions can be *eyeballed*: every clip's label
// says which way the human moves first, and the robot must follow the same
// way (docs/CANONICAL.md; a wrong robot-side sign is fixed with gain: -1 in
// its ROBOT.md, never here).
const TWO_PI = 2 * Math.PI;

function synth(name, label, seconds, fn, rate = 30) {
  const n = Math.round(seconds * rate) + 1;
  const t = Array.from({ length: n }, (_, i) => i / rate);
  const data = {};
  for (const c of CHANNELS) if (c !== 't') data[c] = new Array(n);
  for (let i = 0; i < n; i++) {
    const f = neutralFrame();
    f.face_valid = 1;
    fn(f, t[i], seconds);
    for (const c of CHANNELS) if (c !== 't') data[c][i] = f[c];
  }
  const tr = new Track({ kind: 'canonical', name, t, data, group: 'calibration' });
  tr.label = label;
  return tr;
}

// half-sine pulse of `width` seconds starting at t0, 0 outside
const pulse = (t, t0, width) => (t < t0 || t > t0 + width ? 0 : Math.sin((Math.PI * (t - t0)) / width));

export function syntheticClips() {
  return [
    synth('cal_look_left_right', 'cal: look LEFT then RIGHT (head_yaw +30 → −30)', 4, (f, t) => {
      f.head_yaw = 30 * Math.sin(TWO_PI * 0.25 * t);
    }),
    synth('cal_look_up_down', 'cal: look UP then DOWN (head_pitch +20 → −20)', 4, (f, t) => {
      f.head_pitch = 20 * Math.sin(TWO_PI * 0.25 * t);
    }),
    synth('cal_roll', 'cal: roll — RIGHT ear down first (head_roll +20 → −20)', 4, (f, t) => {
      f.head_roll = 20 * Math.sin(TWO_PI * 0.25 * t);
    }),
    synth('cal_lean_in', 'cal: lean IN toward camera, twice (head_x +80 mm, torso_lean_fwd +15)', 4, (f, t) => {
      const p = pulse(t, 0.3, 1.5) + pulse(t, 2.3, 1.5);
      f.head_x = 80 * p;
      f.torso_lean_fwd = 15 * p;
    }),
    synth('cal_brows', 'cal: brows — both ×2, then LEFT only, then RIGHT only (brow_l/r 0 → 1)', 5, (f, t) => {
      const both = pulse(t, 0.2, 0.6) + pulse(t, 1.1, 0.6);
      f.brow_l = Math.min(1, both + pulse(t, 2.2, 0.8));
      f.brow_r = Math.min(1, both + pulse(t, 3.4, 0.8));
    }),
    synth('cal_nod', 'cal: nod — small fast head_pitch (±6°, 2.5 Hz)', 3, (f, t) => {
      f.head_pitch = -6 * Math.sin(TWO_PI * 2.5 * t) * (t < 2.4 ? 1 : Math.max(0, (3 - t) / 0.6));
    }),
    synth('cal_talk', 'cal: talk — mouth_open bursts + small head motion, speaking=1 in bursts', 6, (f, t) => {
      const env = Math.max(0, Math.sin(TWO_PI * 0.35 * t)) ** 0.5; // bursts
      const syl = 0.5 + 0.5 * Math.sin(TWO_PI * 4.2 * t + 0.7 * Math.sin(TWO_PI * 1.3 * t));
      f.mouth_open = env > 0.15 ? 0.65 * env * syl : 0;
      f.speaking = env > 0.15 ? 1 : 0;
      f.head_yaw = 4 * Math.sin(TWO_PI * 0.6 * t) * env;
      f.head_pitch = 3 * Math.sin(TWO_PI * 1.1 * t + 1) * env - 2 * env;
      f.head_roll = 2.5 * Math.sin(TWO_PI * 0.45 * t + 2) * env;
      f.brow_l = f.brow_r = 0.25 * env * (0.5 + 0.5 * Math.sin(TWO_PI * 0.8 * t));
    }),
    synth('cal_puppet_wave', 'cal: PUPPET arm wave — shoulder/elbow/wrist sinusoids (arm_valid=1; use mapping = puppet)', 6, (f, t) => {
      f.arm_valid = 1;
      f.shoulder_pitch = 90 + 25 * Math.sin(TWO_PI * 0.3 * t);
      f.shoulder_yaw = 20 * Math.sin(TWO_PI * 0.2 * t);
      f.elbow_flex = 45 + 35 * Math.sin(TWO_PI * 0.6 * t);
      f.wrist_roll = 35 * Math.sin(TWO_PI * 0.5 * t + 1);
      f.wrist_pitch = 25 * Math.sin(TWO_PI * 0.8 * t);
      f.hand_open = 0.5 + 0.5 * Math.sin(TWO_PI * 0.4 * t);
    }),
  ];
}

// ---------------------------------------------------------------------------
// Listing helpers (work on python -m http.server AND on GitHub Pages)
// ---------------------------------------------------------------------------

/** GET JSON, null on any failure (never throws). */
export async function fetchJsonOrNull(url) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

/** GET text, null on any failure (never throws). */
export async function fetchTextOrNull(url) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) return null;
    return await r.text();
  } catch (e) {
    return null;
  }
}

/** True if a URL responds 2xx to a GET (HEAD is not supported everywhere). */
export async function urlExists(url) {
  try {
    const r = await fetch(url, { method: 'GET', cache: 'no-cache' });
    return r.ok;
  } catch (e) {
    return false;
  }
}

/** Normalise a clip listing: strings, {file}, {name} or {name, description} → {file, name, description}. */
export function normaliseListing(entries, ext) {
  return (entries || []).map((e) => {
    if (typeof e === 'string') return { file: e, name: e.replace(/\.[^.]+$/, ''), description: '' };
    const file = e.file || e.path || (e.name ? e.name + ext : null);
    const desc = e.description || [e.title, e.license && `(${e.license})`, e.seconds && `${Math.round(e.seconds)} s`].filter(Boolean).join(' ');
    return { file, name: e.name || (file || '').replace(/\.[^.]+$/, ''), description: desc };
  }).filter((e) => e.file && e.file.endsWith(ext) && e.file !== 'index.json');
}

/**
 * List files with `ext` in a directory URL. Every probe is optional (a
 * missing file is a 404 in the console), so callers say which to try:
 *   index:   <dir>/index.json  — ["a.csv", ...] or [{file}|{name}, ...] or {files|clips:[...]}
 *   listing: the server's HTML directory listing (python -m http.server; not GitHub Pages)
 * then `fallback` names.
 * @returns {Promise<Array<{file:string,name:string,description:string}>>}
 */
export async function listDir(dirUrl, ext, { index = true, listing = true, fallback = [] } = {}) {
  if (index) {
    const idx = await fetchJsonOrNull(`${dirUrl}/index.json`);
    if (idx) {
      const arr = Array.isArray(idx) ? idx : idx.files || idx.clips || [];
      const out = normaliseListing(arr, ext);
      if (out.length) return out;
    }
  }
  if (listing) {
    const html = await fetchTextOrNull(`${dirUrl}/`);
    if (html && /<a href=/i.test(html)) {
      const files = [];
      const re = /href="([^"]+)"/gi;
      let m;
      while ((m = re.exec(html))) {
        const f = decodeURIComponent(m[1]);
        if (f.endsWith(ext) && !f.includes('/')) files.push(f);
      }
      if (files.length) return normaliseListing(files.sort(), ext);
    }
  }
  return normaliseListing(fallback.map((n) => (n.endsWith(ext) ? n : n + ext)), ext);
}
