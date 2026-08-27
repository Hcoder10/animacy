// animacy browser viewer: two URDF robots driven by one canonical motion space.
//
//   profile JSON (web/robots/<name>.json, written by `animacy profile export`)
//     └─ description.urdf → ../robots/<name>/<urdf>  → RobotViewer (three.js + urdf-loader)
//     └─ retarget.<mode>  → LiveRetargeter (same equations as animacy/retarget.py)
//   motion source (motion_source.js) → frame → retargeter → toUrdfValues → viewer.setJoints
//
// Everything that can fail (fetches, camera, models) is caught and shown in
// the status line; the render loop never throws.

import { RobotViewer } from './viewer.js';
import { LiveRetargeter, toUrdfValues, restValues } from './retarget.js';
import { CHANNELS, BOUNDS, UNITS, FLAGS } from './canonical.js';
import { parseAutonomousCsv, parseJointJson, parseCanonicalJson, syntheticClips, listDir, normaliseListing, fetchJsonOrNull, fetchTextOrNull, urlExists } from './clips.js';
import { ClipSource, SyntheticSource, WebcamSource } from './motion_source.js';
import { TalkSource, ListenSource, MotionBackends, KOKORO_VOICES } from './talk.js';
import { Recorder, SessionRunner } from './record.js';
import { STANDINS } from './standins.js';

// ---------------------------------------------------------------------------
// pins (also listed in web/README.md)
// ---------------------------------------------------------------------------
const MP_VERSION = '0.10.21';
const CDN = {
  vision: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  faceModel: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  poseModel: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
};

// The headline layout: these two open at boot. Every other robot in
// web/manifest.json (any robots/<name>/ROBOT.md exported to web/robots/<name>.json)
// is offered by the "+ add robot" picker (or `?robots=lamp,reachy_mini,so101`).
const HEADLINE_ROBOTS = ['lamp', 'reachy_mini'];
// GitHub Pages has no directory listing; these are the 31 Autonomous OS recordings.
const LAMP_NATIVE_FALLBACK = [
  'acknowledge', 'confused', 'curious', 'excited', 'fear', 'goodbye', 'greeting', 'happy_wiggle', 'headshake', 'idle',
  'laugh', 'listening', 'music_chill', 'music_classical', 'music_groove', 'music_hiphop', 'music_hype', 'music_jazz',
  'music_rock', 'music_waltz', 'nod', 'playful', 'sad', 'scanning', 'shock', 'shy', 'sleepy', 'stretching',
  'thinking_deep', 'tracking', 'wake_up',
];
const READOUT_CHANNELS = CHANNELS.filter((c) => c !== 't');

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const IS_LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);
// web/manifest.json (python web/dev/build_manifest.py) says what exists so the
// static site never probes for files that may 404.
let manifest = null;

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const app = {
  ready: false,
  robotNames: [],        // loaded robots, in viewport order
  robots: {},            // name → {profile, viewer, retargeters, alias, standin, urdfUrl, values, urdf, missingJoints}
  sourceKind: null,      // 'native' | 'canonical' | 'webcam' | 'model'
  source: null,
  clips: { native: [], canonical: [] },
  clipId: null,
  mode: 'default',
  channels: null,        // latest canonical frame
  ab: { on: false, viewer: null, source: null, clipId: null, values: null },
  fps: 0,
  errors: [],
  webcam: null,
  loop: true,
  speed: 1.0,
  backends: null,        // MotionBackends (model / retrieval / envelope), built from manifest.bundle
  talk: null,            // TalkSource while the Talk tab is active
  recorder: null,        // Recorder while Webcam live is active
  session: null,         // SessionRunner while a guided session runs
};
window.animacy = app; // test / debugging API, extended at the bottom

// ---------------------------------------------------------------------------
// status + errors
// ---------------------------------------------------------------------------
function setStatus(msg, kind = '') {
  const el = $('status');
  el.textContent = msg;
  el.className = `status ${kind}`;
}
let toastTimer = null;
function toast(msg, ms = 6000) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
}
function reportError(where, err) {
  const msg = `${where}: ${err && err.message ? err.message : err}`;
  app.errors.push(msg);
  console.warn('[animacy]', msg, err);
  setStatus(msg, 'err');
  toast(msg);
}
window.addEventListener('error', (e) => { app.errors.push(String(e.message)); });
window.addEventListener('unhandledrejection', (e) => { reportError('promise', e.reason); });

// ---------------------------------------------------------------------------
// robots
// ---------------------------------------------------------------------------
async function loadRobot(name, container, { forAb = false } = {}) {
  const profile = await fetchJsonOrNull(`robots/${name}.json`);
  if (!profile) throw new Error(`robots/${name}.json missing — run: animacy profile export robots/${name} -o web/robots/${name}.json`);
  const robotDir = `../robots/${name}`;
  let urdfUrl = `${robotDir}/${profile.description.urdf}`;
  // package://<x>/ inside a shipped URDF resolves inside the robot folder
  let packages = (pkg) => (pkg === name ? robotDir : `${robotDir}/${pkg}`);
  let alias = null;
  let standin = null;
  let anchorLink = null;
  let cameraDistance = (profile.description.viewer && profile.description.viewer.camera_distance) || null;

  const override = params.get(`urdf_${name}`);
  const mrobot = manifest && manifest.robots && manifest.robots[name];
  const urdfExists = async () => (mrobot ? mrobot.exists && mrobot.urdf === profile.description.urdf : urlExists(urdfUrl));
  if (override) {
    urdfUrl = override;
    packages = (pkg) => `${override.replace(/\/[^/]*$/, '')}/${pkg}`;
  } else if (params.get('standin') === '1' || !(await urdfExists())) {
    standin = STANDINS[name] || null;
    if (standin) {
      urdfUrl = standin.urdf;
      packages = standin.packages;
      alias = standin.joints;
      anchorLink = standin.anchorLink || null;
      cameraDistance = standin.cameraDistance || cameraDistance;
    }
  }

  const viewer = new RobotViewer(container);
  const loadingEl = $(`loading-${forAb ? 'lamp-ab' : name}`);
  try {
    await viewer.loadRobot({ urdfUrl, packages, meshScale: profile.description.mesh_scale || 1, cameraDistance, anchorLink });
    loadingEl.hidden = true;
  } catch (e) {
    loadingEl.textContent = `could not load ${urdfUrl}`;
    loadingEl.classList.add('err');
    throw e;
  }
  const have = new Set(viewer.jointNames);
  const urdfNameOf = (j) => (alias && alias[j.name] && alias[j.name].urdf_joint) || j.urdf_joint || j.name;
  const missingJoints = profile.joints.filter((j) => !have.has(urdfNameOf(j))).map((j) => j.name);
  const retargeters = {};
  for (const mode of Object.keys(profile.retarget)) retargeters[mode] = new LiveRetargeter(profile, mode);
  const R = { name, profile, viewer, retargeters, alias, standin, urdfUrl, values: restValues(profile), urdf: {}, missingJoints };
  applyJoints(R, R.values);
  return R;
}

function applyJoints(R, values) {
  R.values = values;
  R.urdf = toUrdfValues(values, R.profile, R.alias);
  R.viewer.setJoints(R.urdf);
}

function restAll(except = null) {
  for (const n of app.robotNames) {
    if (n === except) continue;
    const R = app.robots[n];
    if (!R) continue;
    for (const rt of Object.values(R.retargeters)) rt.reset();
    applyJoints(R, restValues(R.profile));
  }
}

// ---------------------------------------------------------------------------
// viewports + joint panels are created per robot (the set comes from the manifest)
// ---------------------------------------------------------------------------
function manifestRobots() {
  const names = manifest && manifest.robots ? Object.keys(manifest.robots) : [];
  return names.length ? names : HEADLINE_ROBOTS.slice();
}

function robotLabel(name) {
  const m = manifest && manifest.robots && manifest.robots[name];
  return (m && m.display_name) || (app.robots[name] && app.robots[name].profile.display_name) || name;
}

function createViewport(name, { before = null, removable = false } = {}) {
  const sec = document.createElement('section');
  sec.className = 'viewport';
  sec.id = `vp-${name}`;
  sec.dataset.robot = name;
  sec.innerHTML = `<div class="vp-label"><span class="vp-title">${robotLabel(name)}</span><span class="badge" id="badge-${name}"></span></div>` +
    `<div class="vp-sub" id="sub-${name}"></div><div class="vp-loading" id="loading-${name}">loading URDF…</div>` +
    (removable ? `<button class="vp-close" title="close this viewport">✕</button>` : '');
  const parent = $('viewports');
  if (before) parent.insertBefore(sec, before); else parent.insertBefore(sec, $('webcam-thumb'));
  if (removable) sec.querySelector('.vp-close').addEventListener('click', () => removeRobot(name));
  return sec;
}

function createJointPanel(R) {
  const units = [...new Set(R.profile.joints.map((j) => j.unit))].join(' / ');
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.id = `panel-${R.name}`;
  panel.innerHTML = `<h3>${robotLabel(R.name)} joints <span class="muted">${units}</span></h3><div class="bars" id="joint-bars-${R.name}"></div>`;
  document.querySelector('.readouts').appendChild(panel);
  const jp = panel.querySelector('.bars');
  jp.style.gridTemplateRows = `repeat(${Math.min(7, R.profile.joints.length)}, 18px)`;
  bars.joints[R.name] = {};
  for (const j of R.profile.joints) {
    const b = makeBar(jp, j.name, j.name, 'joint');
    bars.joints[R.name][j.name] = { ...b, lo: j.min, hi: j.max, signed: j.min < 0 && j.max > 0 };
    if (!(j.min < 0 && j.max > 0)) b.mid.style.display = 'none';
  }
  layoutReadouts();
}

// Two robots: 4 channel columns + 2-column joint panels in a 7-row strip. More
// robots: narrower panels, so 3 channel columns and single-column joint panels
// (9 rows); the strip grows a little instead of overflowing.
function layoutReadouts() {
  const n = app.robotNames.length;
  const many = n > 2;
  const ro = document.querySelector('.readouts');
  ro.classList.toggle('many', many);
  ro.style.gridTemplateColumns = `${many ? 1.6 : 2.1}fr repeat(${Math.max(1, n)}, 1fr)`;
  const rows = many ? 9 : 7;
  $('channel-bars').style.gridTemplateRows = `repeat(${Math.ceil(READOUT_CHANNELS.length / (many ? 3 : 4))}, 18px)`;
  for (const name of app.robotNames) {
    const jp = $(`joint-bars-${name}`);
    if (jp) jp.style.gridTemplateRows = `repeat(${Math.min(rows, app.robots[name].profile.joints.length)}, 18px)`;
  }
}

/** Open a viewport for `name` (from the manifest) and drive it with the current source. */
async function addRobot(name, { headline = false, before = null } = {}) {
  if (app.robots[name]) return app.robots[name];
  const sec = createViewport(name, { before, removable: !headline });
  let R;
  try {
    R = await loadRobot(name, sec);
  } catch (e) {
    sec.remove();
    reportError(`load ${name}`, e);
    throw e;
  }
  app.robots[name] = R;
  app.robotNames.push(name);
  describeRobot(R);
  createJointPanel(R);
  fillModeSelect();
  fillAddRobotPicker();
  if (app.sourceKind === 'native') applyJoints(R, restValues(R.profile));
  return R;
}

function removeRobot(name) {
  if (HEADLINE_ROBOTS.includes(name) || !app.robots[name]) return;
  const R = app.robots[name];
  R.viewer.dispose();
  const sec = $(`vp-${name}`);
  if (sec) sec.remove();
  const panel = $(`panel-${name}`);
  if (panel) panel.remove();
  delete bars.joints[name];
  delete app.robots[name];
  app.robotNames = app.robotNames.filter((n) => n !== name);
  layoutReadouts();
  fillModeSelect();
  fillAddRobotPicker();
}

function fillAddRobotPicker() {
  const sel = $('add-robot');
  sel.innerHTML = '<option value="">+ add robot…</option>';
  for (const name of manifestRobots()) {
    if (app.robots[name]) continue;
    const m = manifest && manifest.robots && manifest.robots[name];
    const o = document.createElement('option');
    o.value = name;
    o.textContent = `${robotLabel(name)}${m && m.exists === false ? ' (no URDF)' : ''}`;
    sel.appendChild(o);
  }
  sel.hidden = sel.options.length <= 1;
}

/** Retarget modes = the union over loaded robots; a robot lacking the chosen mode uses its `default`. */
function fillModeSelect() {
  const labels = { default: 'default (face + torso)', puppet: 'puppet (your arm)' };
  const modes = [];
  for (const n of app.robotNames) for (const m of Object.keys(app.robots[n].profile.retarget)) if (!modes.includes(m)) modes.push(m);
  if (!modes.length) modes.push('default');
  const sel = $('mode-select');
  const cur = sel.value || app.mode;
  sel.innerHTML = '';
  for (const m of modes) {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = labels[m] || m;
    sel.appendChild(o);
  }
  sel.value = modes.includes(cur) ? cur : modes[0];
}

function describeRobot(R, forAb = false) {
  const id = forAb ? 'lamp-ab' : R.name;
  const badge = $(`badge-${id}`);
  const sub = $(`sub-${id}`);
  if (R.standin) {
    badge.textContent = 'stand-in URDF';
    badge.className = 'badge warn';
    if (!forAb) sub.textContent = R.standin.note + (R.missingJoints.length ? ` · not driven: ${R.missingJoints.join(', ')}` : '');
  } else {
    badge.textContent = R.missingJoints.length ? `${R.missingJoints.length} joint(s) missing` : '';
    badge.className = R.missingJoints.length ? 'badge warn' : 'badge';
    if (!forAb) {
      const p = R.profile;
      sub.textContent = `${p.vendor} · ${p.joints.length} joints · ${R.profile.description.urdf}` + (R.missingJoints.length ? ` · missing in URDF: ${R.missingJoints.join(', ')}` : '');
    }
  }
}

// ---------------------------------------------------------------------------
// clips
// ---------------------------------------------------------------------------
const trackCache = new Map();

async function discoverClips() {
  // native clips: one list per robot from the manifest (format by extension: Autonomous CSV
  // or animacy joint-table JSON); without a manifest, probe the two headline robots
  const addNative = (robot, files) => {
    for (const f of files) {
      const fmt = f.file.endsWith('.csv') ? 'autonomous_csv' : 'joint_json';
      app.clips.native.push({ id: `${robot}/${f.name}`, label: `${robot} · ${f.name}`, robot, url: `../robots/${robot}/clips/native/${f.file}`, fmt, description: f.description });
    }
  };
  if (manifest && manifest.native) {
    for (const robot of Object.keys(manifest.native)) {
      addNative(robot, [...normaliseListing(manifest.native[robot], '.csv'), ...normaliseListing(manifest.native[robot], '.json')]);
    }
  } else {
    addNative('lamp', await listDir('../robots/lamp/clips/native', '.csv', { fallback: LAMP_NATIVE_FALLBACK }));
    addNative('reachy_mini', await listDir('../robots/reachy_mini/clips/native', '.json', { fallback: [] }));
  }
  // canonical: synthetic calibration + captured (web/clips/*.json)
  for (const tr of syntheticClips()) {
    trackCache.set(`synth/${tr.name}`, tr);
    app.clips.canonical.push({ id: `synth/${tr.name}`, label: tr.label, group: 'calibration', synthetic: true });
  }
  const seen = new Set();
  const addCaptured = (f) => {
    if (seen.has(f.file)) return;
    seen.add(f.file);
    app.clips.canonical.push({ id: `clip/${f.name}`, label: `captured · ${f.name}`, url: `clips/${f.file}`, group: 'captured', description: f.description });
  };
  if (manifest && manifest.clips) normaliseListing(manifest.clips, '.json').forEach(addCaptured);
  // on a local http.server, newly dropped files show up without rebuilding the manifest
  if (IS_LOCAL || !manifest) (await listDir('clips', '.json', { index: !manifest, listing: true })).forEach(addCaptured);
}

function pollenMoveToTrack(obj, name) {
  // Pollen recorded-move JSON (export.py to_pollen_move) → joint table
  const t = obj.time.map((x) => x - obj.time[0]);
  const data = { head_x: [], head_y: [], head_z: [], head_roll: [], head_pitch: [], head_yaw: [], antenna_left: [], antenna_right: [], body_yaw: [] };
  const RAD = 180 / Math.PI;
  for (const s of obj.set_target_data) {
    const M = s.head;
    data.head_x.push(M[0][3] * 1000); data.head_y.push(M[1][3] * 1000); data.head_z.push(M[2][3] * 1000);
    // ZYX euler from R = Rz·Ry·Rx (matches export._rpy_to_matrix)
    data.head_yaw.push(Math.atan2(M[1][0], M[0][0]) * RAD);
    data.head_pitch.push(Math.asin(Math.min(1, Math.max(-1, -M[2][0]))) * RAD);
    data.head_roll.push(Math.atan2(M[2][1], M[2][2]) * RAD);
    data.antenna_left.push(s.antennas[0] * RAD); data.antenna_right.push(s.antennas[1] * RAD);
    data.body_yaw.push((s.body_yaw || 0) * RAD);
  }
  return parseJointJson({ robot: 'reachy_mini', t, data, joints: Object.keys(data) }, name, 'reachy_mini');
}

async function getTrack(entry) {
  if (trackCache.has(entry.id)) return trackCache.get(entry.id);
  let tr;
  if (entry.fmt === 'autonomous_csv') {
    const text = await fetchTextOrNull(entry.url);
    if (text === null) throw new Error(`clip not found: ${entry.url}`);
    tr = parseAutonomousCsv(text, entry.label, entry.robot);
  } else if (entry.fmt === 'joint_json') {
    const obj = await fetchJsonOrNull(entry.url);
    if (!obj) throw new Error(`clip not found: ${entry.url}`);
    tr = obj.set_target_data ? pollenMoveToTrack(obj, entry.label) : parseJointJson(obj, entry.label, entry.robot);
  } else {
    const obj = await fetchJsonOrNull(entry.url);
    if (!obj) throw new Error(`clip not found: ${entry.url}`);
    tr = parseCanonicalJson(obj, entry.label);
  }
  trackCache.set(entry.id, tr);
  return tr;
}

function fillClipSelect(kind) {
  const sel = $('clip-select');
  sel.innerHTML = '';
  const list = app.clips[kind] || [];
  if (!list.length) {
    const o = document.createElement('option');
    o.textContent = kind === 'native' ? 'no native clips found' : 'no clips';
    sel.appendChild(o);
    sel.disabled = true;
    return;
  }
  sel.disabled = false;
  let group = null, og = null;
  for (const e of list) {
    const g = e.group || e.robot || '';
    if (g !== group) { og = document.createElement('optgroup'); og.label = g; sel.appendChild(og); group = g; }
    const o = document.createElement('option');
    o.value = e.id;
    o.textContent = e.label;
    if (e.description) o.title = e.description;
    (og || sel).appendChild(o);
  }
}

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------
function stopSource() {
  if (app.source && app.source.stop) { try { app.source.stop(); } catch (e) { /* ignore */ } }
  app.source = null;
  app.channels = null;
  $('time').textContent = '0.00 / 0.00 s';
  $('scrub').value = '0';
  $('play').textContent = '▶';
}

async function setSource(kind) {
  if (kind === 'model') kind = 'talk'; // old name
  if (kind === app.sourceKind && kind !== 'webcam' && kind !== 'listen') { return; }
  const prev = app.sourceKind;
  stopSource();
  app.talk = null;
  app.sourceKind = kind;
  for (const b of $('source-tabs').querySelectorAll('button')) b.classList.toggle('active', b.dataset.source === kind);
  const isClip = kind === 'native' || kind === 'canonical';
  $('clip-select').hidden = !isClip;
  $('webcam-controls').hidden = kind !== 'webcam';
  $('webcam-thumb').hidden = kind !== 'webcam' && kind !== 'listen';
  $('talk-controls').hidden = kind !== 'talk';
  $('listen-controls').hidden = kind !== 'listen';
  $('mode-select').disabled = kind === 'native';
  for (const id of ['play', 'loop', 'scrub']) $(id).disabled = !(isClip || kind === 'talk');
  $('speed').disabled = !isClip;
  restAll();
  try {
    if (isClip) {
      fillClipSelect(kind);
      const want = app.clipId && app.clips[kind].some((e) => e.id === app.clipId) ? app.clipId : (app.clips[kind][0] && app.clips[kind][0].id);
      if (want) await setClip(want);
      else setStatus(kind === 'native' ? 'no native clips found' : 'no canonical clips', 'err');
    } else if (kind === 'webcam') {
      await startWebcam();
    } else if (kind === 'talk') {
      const t = new TalkSource({ backends: app.backends, backend: $('talk-backend').value, onStatus: talkStatus, loop: app.loop });
      app.source = t;
      app.talk = t;
      const avail = app.backends.available();
      setStatus(`talk: type a line and press Say it · backends: ${avail.join(' / ')}${app.backends.hasModel ? '' : ' (no model bundle in web/models/ — envelope heuristic only)'}`, 'ok');
    } else if (kind === 'listen') {
      const l = new ListenSource({
        backends: app.backends, backend: $('listen-backend').value,
        video: $('webcam-video'), overlay: $('webcam-overlay'), cdn: CDN,
        onStatus: (m) => { $('listen-stat').textContent = m; $('webcam-status').textContent = m; setStatus(`listen: ${m}`); },
      });
      app.source = l;
      await l.start();
      setStatus('listen: microphone → VAD → causal model (speaking = 0) + gaze overlay from the camera → both robots · experimental', 'ok');
    }
  } catch (e) {
    reportError(`source ${kind}`, e);
    if ((kind === 'webcam' || kind === 'listen') && prev && prev !== kind) { await setSource(prev); }
  }
}

function talkStatus(msg, frac) {
  const bar = $('talk-progress');
  if (frac !== undefined && frac !== null && frac < 1) {
    bar.hidden = false;
    bar.querySelector('.fill').style.width = `${Math.round(frac * 100)}%`;
  } else {
    bar.hidden = true;
  }
  setStatus(`talk: ${msg}`, frac === 1 ? 'ok' : '');
}

async function sayText(text) {
  if (!app.talk) await setSource('talk');
  const btn = $('talk-say');
  btn.disabled = true;
  try {
    app.talk.backend = $('talk-backend').value;
    const info = await app.talk.say(text, { voice: $('talk-voice').value });
    $('scrub').max = String(app.talk.duration || 1);
    return info;
  } catch (e) {
    reportError('talk', e);
    return null;
  } finally {
    btn.disabled = false;
  }
}

/** Test/dev entry: animate a waveform without TTS (see web/dev/screenshot.py). */
async function sayAudio(samples, sr, backend = null) {
  if (!app.talk) await setSource('talk');
  if (backend) { app.talk.backend = backend; $('talk-backend').value = backend; }
  const info = await app.talk.sayAudio(Float32Array.from(samples), sr);
  $('scrub').max = String(app.talk.duration || 1);
  return info;
}

function fillBackendSelects() {
  const avail = app.backends ? app.backends.available() : ['envelope'];
  const labels = {
    model: 'model (learned, ONNX) — experimental',
    retrieval: 'retrieval (motion matching, default)',
    envelope: 'envelope heuristic (no model)',
  };
  const titles = {
    model: 'learned audio→motion (VQ codes + transformer). Experimental: beats its shuffled-audio control on one of two held-out speakers — see docs/RESULTS.md',
    retrieval: 'motion matching: real human motion from the corpus, aligned to the speech (the default source)',
    envelope: 'explicit speech-energy heuristic, labelled as such; the floor the learned sources must beat',
  };
  for (const id of ['talk-backend', 'listen-backend']) {
    const sel = $(id);
    sel.innerHTML = '';
    for (const b of avail) {
      const o = document.createElement('option');
      o.value = b;
      o.textContent = labels[b] || b;
      if (titles[b]) o.title = titles[b];
      sel.appendChild(o);
    }
    sel.value = avail[0];
  }
  const vs = $('talk-voice');
  vs.innerHTML = '';
  for (const v of KOKORO_VOICES) {
    const o = document.createElement('option');
    o.value = v;
    o.textContent = v;
    vs.appendChild(o);
  }
}

async function setClip(id) {
  const kind = app.sourceKind;
  const entry = (app.clips[kind] || []).find((e) => e.id === id);
  if (!entry) throw new Error(`unknown clip ${id}`);
  app.clipId = id;
  $('clip-select').value = id;
  setStatus(`loading ${entry.label}…`);
  const track = await getTrack(entry);
  stopSource();
  const Src = entry.synthetic ? SyntheticSource : ClipSource;
  app.source = new Src(track, { loop: app.loop, speed: app.speed });
  if (params.get('autoplay') === '0') app.source.pause();
  if (track.kind === 'joint') restAll(track.robot);
  else for (const R of Object.values(app.robots)) for (const rt of Object.values(R.retargeters)) rt.reset();
  $('scrub').max = String(track.duration || 1);
  $('play').textContent = app.source.playing ? '❚❚' : '▶';
  const info = track.kind === 'joint'
    ? `${entry.label} — vendor clip, ${track.n} frames, ${track.duration.toFixed(2)} s, plays raw on ${track.robot}`
    : `${entry.label} — canonical, ${track.n} frames @ ${track.duration.toFixed(2)} s → retargeted through both ROBOT.md (${app.mode})`;
  setStatus(info, 'ok');
}

async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('this browser has no getUserMedia (serve over http://localhost or https)');
  const wc = new WebcamSource({
    video: $('webcam-video'),
    overlay: $('webcam-overlay'),
    arm: $('arm-select').value,
    onStatus: (m) => { $('webcam-status').textContent = m; setStatus(`webcam: ${m}`); },
    cdn: CDN,
  });
  app.source = wc;
  app.webcam = wc;
  await wc.start();
  for (const R of Object.values(app.robots)) for (const rt of Object.values(R.retargeters)) rt.reset();
  app.recorder = new Recorder({ webcam: wc, onStatus: (m) => setStatus(`record: ${m}`, 'ok') });
  wc.onFrame((f) => app.recorder.onFrame(f));
  setStatus('webcam live → canonical channels → both robots', 'ok');
}

// ---------------------------------------------------------------------------
// record mode (Webcam live): single takes + the guided session
// ---------------------------------------------------------------------------
async function toggleRecord() {
  const rec = app.recorder;
  if (!rec) { toast('start Webcam live first'); return; }
  const btn = $('rec-toggle');
  try {
    if (!rec.recording) {
      const subject = $('rec-subject').value || 'me';
      await rec.start({ subject, slug: `take${rec.takes.length + 1}`, role: $('rec-role').value, prompt: '(free take)' });
      btn.textContent = '■ Stop';
      btn.classList.add('on');
      $('rec-dot').hidden = false;
    } else {
      await rec.stop();
      btn.textContent = '● Record';
      btn.classList.remove('on');
      $('rec-dot').hidden = true;
      $('rec-download').disabled = false;
      toast(`saved ${rec.take.name}: ${rec.take.n} frames, ${rec.take.seconds.toFixed(1)} s — press "Download take"`, 6000);
    }
  } catch (e) {
    reportError('record', e);
  }
}

function sessionState(s) {
  const p = s.prompt;
  $('sp-progress').textContent = p ? `${s.index + 1}/${s.total} · ${p.slug} · ${p.role} · ${p.seconds}s` : `${s.done.length}/${s.total} takes`;
  const count = $('sp-count');
  count.classList.toggle('rec', s.phase === 'recording');
  if (s.phase === 'prompt' || s.phase === 'countdown') { $('sp-prompt').textContent = p.text; count.textContent = s.phase === 'countdown' ? (s.remaining > 3 ? `read… ${s.remaining}` : String(s.remaining)) : ''; }
  else if (s.phase === 'recording') { $('sp-prompt').textContent = p.text; count.textContent = `● ${s.remaining}s`; }
  else if (s.phase === 'saved') { count.textContent = 'got it'; }
  else if (s.phase === 'error') { $('sp-prompt').textContent = `could not start recording: ${s.error}`; count.textContent = ''; }
  else if (s.phase === 'complete' || s.phase === 'aborted') {
    $('sp-prompt').textContent = s.phase === 'complete' ? `Session complete: ${s.done.length} takes. Download the zip and send it in (CC-BY-4.0). Thank you.` : `Session stopped after ${s.done.length} take(s).`;
    count.textContent = '';
    $('sp-start').disabled = false; $('sp-skip').disabled = true; $('sp-stop').disabled = true;
    $('sp-download').disabled = !s.done.length;
    $('rec-download').disabled = !app.recorder || !app.recorder.take;
  }
  $('rec-dot').hidden = s.phase !== 'recording';
  const ul = $('sp-takes');
  ul.innerHTML = '';
  for (const t of s.done) {
    const li = document.createElement('li');
    li.textContent = `${t.name} — ${t.n} frames, ${t.seconds.toFixed(1)} s, face ${(t.meta.face_valid_fraction * 100).toFixed(0)}%`;
    ul.appendChild(li);
  }
}

async function startSession() {
  if (!app.recorder) { toast('start Webcam live first'); return; }
  if (app.session && app.session.running) return;
  app.session = new SessionRunner({ recorder: app.recorder, subject: $('rec-subject').value || 'me', quick: $('sp-quick').checked, speak: $('sp-speak').checked, onState: sessionState });
  $('sp-start').disabled = true; $('sp-skip').disabled = false; $('sp-stop').disabled = false; $('sp-download').disabled = true;
  try { await app.session.run(); } catch (e) { reportError('session', e); }
}

function wireRecordUi() {
  $('rec-toggle').addEventListener('click', toggleRecord);
  $('rec-download').addEventListener('click', () => { if (app.recorder) app.recorder.download(); });
  $('rec-session').addEventListener('click', () => { $('session-panel').hidden = false; });
  $('sp-close').addEventListener('click', () => { if (app.session && app.session.running) app.session.abort = true; $('session-panel').hidden = true; });
  $('sp-start').addEventListener('click', startSession);
  $('sp-skip').addEventListener('click', () => { if (app.session) app.session.skip = true; });
  $('sp-stop').addEventListener('click', () => { if (app.session) app.session.abort = true; });
  $('sp-download').addEventListener('click', () => { if (app.recorder) app.recorder.downloadSession($('rec-subject').value); });
}

function setMode(mode) {
  app.mode = mode;
  $('mode-select').value = mode;
  for (const R of Object.values(app.robots)) {
    if (!R.retargeters[mode]) continue;
    // continue from the current pose instead of snapping to rest
    Object.assign(R.retargeters[mode].state, R.values);
  }
  const hint = mode === 'puppet' ? ' · puppet: your arm drives the lamp; wrist drives the reachy head' : '';
  if (app.sourceKind !== 'native') setStatus(`mapping = ${mode}${hint}`, 'ok');
}

// ---------------------------------------------------------------------------
// A/B: a second lamp playing the vendor's raw clip
// ---------------------------------------------------------------------------
async function setAb(on) {
  app.ab.on = on;
  $('ab').checked = on;
  $('ab-select').disabled = !on;
  const vp = $('vp-lamp-ab');
  vp.hidden = !on;
  if (!on) { if (app.ab.source) app.ab.source.pause(); return; }
  try {
    if (!app.ab.viewer) {
      const R = await loadRobot('lamp', vp, { forAb: true });
      app.ab.viewer = R;
      describeRobot(R, true);
    }
    const id = app.ab.clipId || (app.clips.native.find((e) => e.robot === 'lamp') || {}).id;
    if (id) await setAbClip(id);
  } catch (e) {
    reportError('A/B', e);
    app.ab.on = false; $('ab').checked = false; vp.hidden = true;
  }
}

async function setAbClip(id) {
  const entry = app.clips.native.find((e) => e.id === id && e.robot === 'lamp');
  if (!entry) return;
  app.ab.clipId = id;
  $('ab-select').value = id;
  const track = await getTrack(entry);
  app.ab.source = new ClipSource(track, { loop: true, speed: app.speed });
  $('sub-lamp-ab').textContent = `B = vendor's hand-authored '${entry.label.replace('lamp · ', '')}' CSV, played raw · A (left) = retargeted from the human clip`;
}

// ---------------------------------------------------------------------------
// readouts
// ---------------------------------------------------------------------------
const bars = { channels: {}, joints: {} };

function makeBar(parent, key, label, cls = '') {
  const el = document.createElement('div');
  el.className = `bar ${cls}`;
  el.innerHTML = `<span class="k" title="${key}">${label}</span><span class="track"><span class="mid"></span><span class="fill"></span></span><span class="v">–</span>`;
  parent.appendChild(el);
  return { el, fill: el.querySelector('.fill'), v: el.querySelector('.v'), mid: el.querySelector('.mid') };
}

function buildReadouts() {
  const cp = $('channel-bars');
  cp.style.gridTemplateRows = `repeat(${Math.ceil(READOUT_CHANNELS.length / 4)}, 18px)`;
  for (const c of READOUT_CHANNELS) {
    const [lo, hi] = BOUNDS[c];
    bars.channels[c] = { ...makeBar(cp, c, c, FLAGS.includes(c) ? 'flag' : ''), lo, hi, signed: lo < 0 };
    if (!(lo < 0)) bars.channels[c].mid.style.display = 'none';
  }
}

function setBar(b, v, unit = '') {
  if (v === null || v === undefined || Number.isNaN(v)) { b.fill.style.width = '0%'; b.v.textContent = '–'; b.el.classList.add('stale'); return; }
  b.el.classList.remove('stale');
  const span = b.hi - b.lo || 1;
  if (b.signed) {
    const f = Math.max(-1, Math.min(1, v / Math.max(Math.abs(b.lo), Math.abs(b.hi))));
    const w = Math.abs(f) * 50;
    b.fill.style.left = f < 0 ? `${50 - w}%` : '50%';
    b.fill.style.width = `${w}%`;
  } else {
    const f = Math.max(0, Math.min(1, (v - b.lo) / span));
    b.fill.style.left = '0';
    b.fill.style.width = `${f * 100}%`;
  }
  b.v.textContent = unit === 'unit' || unit === 'flag' ? v.toFixed(2) : v.toFixed(1);
}

function updateReadouts() {
  const ch = app.channels;
  for (const c of READOUT_CHANNELS) setBar(bars.channels[c], ch ? ch[c] : null, UNITS[c]);
  $('channels-note').textContent = app.sourceKind === 'native' ? '— (native clip: robot joint space, no canonical frame)' : '— docs/CANONICAL.md, neutral-relative';
  for (const n of app.robotNames) {
    const R = app.robots[n];
    if (!R || !bars.joints[n]) continue;
    const mp = R.profile.retarget[app.mode] || {};
    for (const j of R.profile.joints) {
      const b = bars.joints[n][j.name];
      setBar(b, R.values[j.name]);
      b.el.classList.toggle('unmapped', app.sourceKind !== 'native' && !mp[j.name]);
    }
  }
}

// ---------------------------------------------------------------------------
// frame application + render loop
// ---------------------------------------------------------------------------
// The retargeter is a fixed-step integrator (the robots consume 30 Hz frames and
// the spring tracker is a semi-implicit Euler step): never hand it more than
// one nominal frame at a time. Longer gaps are split into equal sub-steps with
// the same channels; a seek is settled by a second of nominal steps so the
// pose lands where the timeline says without slewing from wherever it was.
const RT_DT_MAX = 1 / 30;
const RT_SEEK_SETTLE_STEPS = 30;

function stepRetargeter(rt, channels, dt, seek) {
  if (seek) {
    let out;
    for (let i = 0; i < RT_SEEK_SETTLE_STEPS; i++) out = rt.step(channels, RT_DT_MAX);
    return out;
  }
  const n = Math.max(1, Math.ceil(dt / RT_DT_MAX));
  let out;
  for (let i = 0; i < n; i++) out = rt.step(channels, dt / n);
  return out;
}

function applyFrame(frame) {
  if (frame.channels) {
    app.channels = frame.channels;
    for (const n of app.robotNames) {
      const R = app.robots[n];
      if (!R) continue;
      const rt = R.retargeters[app.mode] || R.retargeters.default;
      applyJoints(R, stepRetargeter(rt, frame.channels, frame.dt, frame.seek));
    }
  } else if (frame.joints) {
    app.channels = null;
    for (const n in frame.joints) {
      const R = app.robots[n];
      if (!R) continue;
      const vals = { ...restValues(R.profile) };
      for (const k in frame.joints[n]) if (k in vals && !Number.isNaN(frame.joints[n][k])) vals[k] = frame.joints[n][k];
      applyJoints(R, vals);
    }
  }
}

let lastNow = performance.now();
let fpsAcc = { n: 0, t: performance.now() };
let readoutAcc = 0;

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, Math.max(0, (now - lastNow) / 1000));
  lastNow = now;
  try {
    if (app.source) {
      const frame = app.source.update(dt);
      if (frame) applyFrame(frame);
      if (app.source.duration !== undefined) {
        const s = app.source;
        if (document.activeElement !== $('scrub')) $('scrub').value = String(s.time);
        $('time').textContent = `${s.time.toFixed(2)} / ${s.duration.toFixed(2)} s`;
        const glyph = s.playing ? '❚❚' : '▶';
        if ($('play').textContent !== glyph) $('play').textContent = glyph;
      }
      if (app.sourceKind === 'webcam' && app.webcam) {
        const w = app.webcam;
        $('webcam-stat').textContent = `${w.fps.toFixed(0)} fps · face ${w.stats.faceMs.toFixed(0)} ms · pose ${w.stats.poseMs.toFixed(0)} ms · ${w.hasFace ? 'face' : 'no face'} · ${w.hasPose ? 'pose' : 'no pose'}${w.speaking ? ' · speaking' : ''}`;
        if (app.recorder && app.recorder.recording) $('rec-time').textContent = app.recorder.seconds.toFixed(1);
      } else if (app.sourceKind === 'listen' && app.source instanceof ListenSource) {
        const l = app.source;
        $('listen-stat').textContent = `${l.talking ? 'you are talking' : 'quiet'} · queue ${l.queue.length} · gaze ${l.cam ? (l.gaze.hasFace ? `yaw ${l.gaze.yaw.toFixed(0)}° pitch ${l.gaze.pitch.toFixed(0)}°` : 'no face') : 'off (no camera)'}`;
      }
    }
    if (app.ab.on && app.ab.source && app.ab.viewer) {
      if (app.source && app.source.duration !== undefined) app.ab.source.playing = app.source.playing;
      const f = app.ab.source.update(dt);
      if (f && f.joints && f.joints.lamp) {
        const R = app.ab.viewer;
        const vals = { ...restValues(R.profile), ...f.joints.lamp };
        applyJoints(R, vals);
      }
      app.ab.viewer.viewer.render();
    }
    for (const n of app.robotNames) if (app.robots[n]) app.robots[n].viewer.render();
    readoutAcc += dt;
    if (readoutAcc >= 0.05) { readoutAcc = 0; updateReadouts(); }
    fpsAcc.n++;
    if (now - fpsAcc.t >= 1000) {
      app.fps = (fpsAcc.n * 1000) / (now - fpsAcc.t);
      $('fps').textContent = `${app.fps.toFixed(0)} fps`;
      fpsAcc = { n: 0, t: now };
    }
  } catch (e) {
    reportError('frame', e);
  }
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
function wireUi() {
  $('source-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-source]');
    if (b) setSource(b.dataset.source).catch((err) => reportError('source', err));
  });
  $('clip-select').addEventListener('change', (e) => setClip(e.target.value).catch((err) => reportError('clip', err)));
  $('mode-select').addEventListener('change', (e) => setMode(e.target.value));
  $('play').addEventListener('click', () => { if (app.source && app.source.toggle) app.source.toggle(); });
  $('loop').addEventListener('change', (e) => { app.loop = e.target.checked; if (app.source && 'loop' in app.source) app.source.loop = app.loop; });
  $('scrub').addEventListener('input', (e) => { if (app.source && app.source.seek) app.source.seek(parseFloat(e.target.value)); });
  $('speed').addEventListener('input', (e) => {
    app.speed = parseFloat(e.target.value);
    $('speed-val').textContent = `${app.speed.toFixed(2)}×`;
    if (app.source && 'speed' in app.source) app.source.speed = app.speed;
    if (app.ab.source) app.ab.source.speed = app.speed;
  });
  $('ab').addEventListener('change', (e) => setAb(e.target.checked).catch((err) => reportError('A/B', err)));
  $('ab-select').addEventListener('change', (e) => setAbClip(e.target.value).catch((err) => reportError('A/B clip', err)));
  $('set-neutral').addEventListener('click', () => { if (app.webcam) { if (!app.webcam.setNeutral()) toast('no face tracked yet — look at the camera'); } });
  $('arm-select').addEventListener('change', (e) => { if (app.webcam) app.webcam.arm = e.target.value; });
  wireRecordUi();
  $('talk-say').addEventListener('click', () => sayText($('talk-text').value));
  $('talk-text').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); sayText($('talk-text').value); } });
  $('talk-backend').addEventListener('change', (e) => { if (app.talk) app.talk.backend = e.target.value; });
  $('listen-backend').addEventListener('change', (e) => { if (app.source && app.source instanceof ListenSource) app.source.backend = e.target.value; });
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      e.preventDefault();
      if (app.source && app.source.toggle) app.source.toggle();
    }
  });
  // A/B select: lamp native clips only
  const abSel = $('ab-select');
  abSel.innerHTML = '';
  for (const e of app.clips.native.filter((c) => c.robot === 'lamp')) {
    const o = document.createElement('option');
    o.value = e.id;
    o.textContent = e.label.replace('lamp · ', 'B: ');
    abSel.appendChild(o);
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  setStatus('loading robots…');
  manifest = await fetchJsonOrNull('manifest.json');
  if (!manifest) console.warn('[animacy] web/manifest.json missing — probing for files instead (run python web/dev/build_manifest.py)');
  buildReadouts();
  // headline pair first (lamp left of the A/B slot, reachy right of it), then any ?robots=… extras
  const wanted = (params.get('robots') || '').split(',').map((s) => s.trim()).filter(Boolean);
  const headline = HEADLINE_ROBOTS.filter((n) => manifestRobots().includes(n));
  await Promise.allSettled(headline.map((n) => addRobot(n, { headline: true, before: n === 'lamp' ? $('vp-lamp-ab') : null }).catch(() => null)));
  for (const n of wanted) if (!app.robots[n] && manifestRobots().includes(n)) await addRobot(n).catch(() => null);
  await discoverClips();
  app.backends = new MotionBackends({ baseUrl: 'models/', bundle: (manifest && manifest.bundle) || {}, onStatus: talkStatus });
  fillBackendSelects();
  fillModeSelect();
  fillAddRobotPicker();
  wireUi();
  $('add-robot').addEventListener('change', (e) => { const n = e.target.value; e.target.value = ''; if (n) addRobot(n).catch(() => null); });
  requestAnimationFrame((t) => { lastNow = t; fpsAcc.t = t; loop(t); });
  const kind = params.get('source') || 'native';
  if (params.get('clip')) app.clipId = params.get('clip');
  if (params.get('mode')) setMode(params.get('mode'));
  await setSource(kind);
  if (params.get('ab') === '1') await setAb(true);
  app.ready = true;
  if (app.robotNames.length >= headline.length && !app.errors.length) {
    const s = $('status').textContent;
    setStatus(s || 'ready', 'ok');
  }
}

// test / debugging API
Object.assign(app, {
  setSource, setClip, setMode, setAb, setAbClip,
  play: () => app.source && app.source.play && app.source.play(),
  pause: () => app.source && app.source.pause && app.source.pause(),
  seek: (t) => app.source && app.source.seek && app.source.seek(t),
  getJointValues: (name) => (app.robots[name] ? { ...app.robots[name].values } : null),
  getUrdfJoints: (name) => (app.robots[name] ? app.robots[name].viewer.getJoints() : null),
  getChannels: () => (app.channels ? { ...app.channels } : null),
  sourceInfo: () => ({ kind: app.sourceKind, clip: app.clipId, mode: app.mode, time: app.source && app.source.time, duration: app.source && app.source.duration, playing: app.source && app.source.playing }),
  robotInfo: (name) => { const R = app.robots[name]; return R ? { urdfUrl: R.urdfUrl, standin: !!R.standin, missingJoints: R.missingJoints, urdfJoints: R.viewer.jointNames } : null; },
  setView: (kind) => { for (const R of Object.values(app.robots)) R.viewer.setView(kind); if (app.ab.viewer) app.ab.viewer.viewer.setView(kind); },
  addRobot, removeRobot,
  loadedRobots: () => app.robotNames.slice(),   // (app.robotNames itself is the live array)
  manifestRobots,
  say: sayText,
  sayAudio,
  record: {
    start: (o) => app.recorder && app.recorder.start(o),
    stop: () => app.recorder && app.recorder.stop(),
    lastTake: () => (app.recorder && app.recorder.take ? { name: app.recorder.take.name, n: app.recorder.take.n, seconds: app.recorder.take.seconds, meta: app.recorder.take.meta, audioBytes: app.recorder.take.audio.size, zipBytes: app.recorder.take.zip.size, motion: app.recorder.take.motion } : null),
    lastZipBase64: async () => { const t = app.recorder && app.recorder.take; if (!t) return null; const b = new Uint8Array(await t.zip.arrayBuffer()); let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); },
  },
  talkInfo: () => (app.talk ? { last: app.talk.last, backend: app.talk.backend, time: app.talk.time, duration: app.talk.duration, playing: app.talk.playing, available: app.backends.available() } : null),
  // Headless rendering hook (animacy/grade + web/dev/render_clip.py): pose one robot from URDF
  // joint values (radians / metres, keyed by urdf_joint), render once, return the frame as a PNG data URL.
  // Set `animacy.source = null` first so the animation loop stops applying its own frames.
  renderFrame: (name, urdfValues, mime = 'image/png') => {
    const R = app.robots[name];
    if (!R) throw new Error(`no robot '${name}'`);
    if (urdfValues) R.viewer.setJoints(urdfValues);
    R.viewer.render();
    return R.viewer.renderer.domElement.toDataURL(mime);
  },
  // World-space forward (+x) axis of a link, in three.js coordinates: x = robot forward,
  // y = up, z = robot RIGHT (URDF −y). A head turning LEFT makes z go negative; looking UP makes y go positive.
  linkForward: (name, link = 'head') => {
    const R = app.robots[name];
    if (!R || !R.viewer.robot || !R.viewer.robot.links[link]) return null;
    R.viewer.scene.updateMatrixWorld(true);
    const e = R.viewer.robot.links[link].matrixWorld.elements;
    return { x: e[0], y: e[1], z: e[2] };
  },
});

boot().catch((e) => reportError('boot', e));
