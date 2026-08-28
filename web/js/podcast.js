// The podcast set: both URDFs in one three.js scene, driven frame-by-frame from
// data/video/podcast/show.json.
//
// This page has no clock. `window.podcast.seek(i)` applies row i of both robots'
// joint tracks and renders once; the renderer (scripts/video/podcast_render.py)
// calls it for every frame it wants. Frame i of the video is therefore row i of
// the show and sample i/fps*sr of narration.wav, whatever the frame rate of the
// machine doing the drawing.
//
// The URDF load and the joint application are the viewer's, not a copy of it:
// `RobotViewer` (web/js/viewer.js) parses the URDF and owns `setJoints`, and the
// canonical -> URDF unit maths is `toUrdfValues` from web/js/retarget.js. The
// only thing this file does differently is where the robot ends up: instead of
// one robot per scene, both are re-parented into one studio (podcast_set.js).

import * as THREE from 'three';
import { RobotViewer } from './viewer.js';
import { toUrdfValues } from './retarget.js';
import { EYELINE, REACHY_DROP, PLACE, PLINTH_R, buildSet, buildPlinth, cameraRigs, applyRig } from './podcast_set.js';

const params = new URLSearchParams(location.search);
const SHOW_URL = params.get('show') || '../data/video/podcast/show.json';
const CAM = (params.get('cam') || 'A').toUpperCase();
const HOSTS = { lamp: 'lamp', reachy: 'reachy_mini' };   // show.json key -> robots/<name>

const state = {
  show: null, hosts: {}, rigs: null, cam: CAM,
  range: { f0: 0, f1: 0 },     // the take this render covers (camera E pushes across it)
  frame: -1, errors: [],
};

// ---------------------------------------------------------------------------
// renderer + scene
// ---------------------------------------------------------------------------
const canvasHost = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({
  antialias: true, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: true,
});
renderer.setPixelRatio(1);                       // 1 canvas pixel per CSS pixel: the viewport is the frame size
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
canvasHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.02, 40);
buildSet(scene);

function resize() {
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  return { w, h };
}
resize();

// ---------------------------------------------------------------------------
// robots
// ---------------------------------------------------------------------------
/**
 * Load one robot exactly the way the viewer does (profile JSON -> URDF -> meshes),
 * then take the parsed robot out of its own viewport and into the studio scene.
 *
 * The `RobotViewer` instance stays alive and keeps doing the two jobs this page
 * needs from it: it owns the URDF's joint objects and it owns `setJoints`. Its
 * own 1x1 canvas is never drawn.
 */
async function loadHost(key, name) {
  const profile = await (await fetch(`robots/${name}.json`)).json();
  const robotDir = `../robots/${name}`;
  const oven = document.createElement('div');
  oven.className = 'oven';
  document.body.appendChild(oven);
  const rv = new RobotViewer(oven);
  await rv.loadRobot({
    urdfUrl: `${robotDir}/${profile.description.urdf}`,
    packages: (pkg) => (pkg === name ? robotDir : `${robotDir}/${pkg}`),
    meshScale: profile.description.mesh_scale || 1,
  });
  if (!rv.robot) throw new Error(`${name}: URDF did not load`);

  // mount = where the host stands and which way it is turned;
  // zup = the viewer's z-up(URDF) -> y-up(three) rotation, unchanged.
  const mount = new THREE.Group();
  const zup = new THREE.Group();
  zup.rotation.x = -Math.PI / 2;
  mount.add(zup);
  zup.add(rv.robot);
  scene.add(mount);

  rv.robot.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = true;
    o.receiveShadow = true;
    // Keep the viewer's material, calm its specular response. Both shells are
    // faceted STL with rebuilt smooth normals, and a tight highlight under a
    // hard key turns every facet seam into a visible crease.
    o.material.roughness = 0.62;
    o.material.metalness = 0.04;
    o.material.needsUpdate = true;
  });

  const head = rv.robot.links.head || rv.robot;
  return { key, name, profile, rv, robot: rv.robot, mount, head, joints: rv.jointNames };
}

/** Row `i` of a track -> that robot's URDF joint values, through the viewer's own mapping. */
function applyRow(host, track, row) {
  const vals = {};
  for (let k = 0; k < track.joints.length; k++) vals[track.joints[k]] = row[k];
  host.rv.setJoints(toUrdfValues(vals, host.profile));
}

function restRow(key) {
  return state.show.hosts[key].rest;
}

function worldPos(obj) {
  scene.updateMatrixWorld(true);
  return new THREE.Vector3().setFromMatrixPosition(obj.matrixWorld);
}

/**
 * The direction this host is actually looking, in world space. A robot's head
 * link does not have to point where it looks: the lamp's shade axis is
 * `description.viewer.gaze` in the head's own frame (web/robots/lamp.json).
 */
function gazeDir(host) {
  const g = (host.profile.description.viewer && host.profile.description.viewer.gaze) || [1, 0, 0];
  return new THREE.Vector3(g[0], g[1], g[2])
    .transformDirection(host.head.matrixWorld).normalize();
}

/**
 * Place both hosts: turn each toward the other, then solve each plinth height so
 * the two heads land on one eyeline. Measured, not guessed, so it stays right if
 * either URDF changes.
 *
 * Must run with the hosts in their rest pose: the head link's height is a
 * function of the joints, so measuring the URDF's zero pose would put the
 * eyeline wherever the URDF happens to park its arm.
 */
function placeHosts() {
  const plinths = {};
  for (const key of Object.keys(state.hosts)) {
    const h = state.hosts[key];
    const p = PLACE[key];
    h.mount.position.set(p.x, 0, p.z);

    // 1. body yaw: turn the host until its *gaze* points at PLACE.gazeAz. The
    //    lamp's shade axis is not its URDF +x, so this is measured, not assumed.
    h.mount.rotation.y = 0;
    scene.updateMatrixWorld(true);
    const g = gazeDir(h);
    const az0 = Math.atan2(-g.z, g.x);                     // same sense as podcast_set dirAt()
    h.mount.rotation.y = THREE.MathUtils.degToRad(p.gazeAz) - az0;
    scene.updateMatrixWorld(true);

    // 2. plinth height: solve so this head lands on the eyeline
    const headY = worldPos(h.head).y - h.mount.position.y;             // head height above the mount
    const bottom = new THREE.Box3().setFromObject(h.robot).min.y - h.mount.position.y;
    const want = EYELINE - (key === 'reachy' ? REACHY_DROP : 0);
    h.mount.position.y = want - headY;
    const top = Math.max(0.02, h.mount.position.y + bottom);            // where the robot's feet land
    plinths[key] = buildPlinth(scene, p.x, p.z, top, PLINTH_R[key]);
    h.plinthHeight = top;
  }
  scene.updateMatrixWorld(true);
  return plinths;
}

/**
 * The boxes the cameras must cover: each robot's bounding box unioned over the
 * whole show (sampled), plus the union of both. Framing against the range of
 * motion rather than the rest pose is what guarantees nothing swings out of
 * frame mid-take — and the per-host boxes are what the singles are cut to.
 */
function coveredBoxes(samples = 96) {
  const boxes = { all: new THREE.Box3() };
  for (const key of Object.keys(state.hosts)) boxes[key] = new THREE.Box3();
  const n = state.show.n_frames;
  for (let s = 0; s < samples; s++) {
    const i = Math.min(n - 1, Math.round((s * (n - 1)) / Math.max(1, samples - 1)));
    for (const key of Object.keys(state.hosts)) {
      const tr = state.show.tracks[key];
      applyRow(state.hosts[key], tr, tr.values[i]);
    }
    scene.updateMatrixWorld(true);
    for (const key of Object.keys(state.hosts)) {
      boxes[key].expandByObject(state.hosts[key].robot);
      boxes.all.expandByObject(state.hosts[key].robot);
    }
  }
  return boxes;
}

// ---------------------------------------------------------------------------
// player
// ---------------------------------------------------------------------------
function setCamera(name, f0 = null, f1 = null) {
  const k = String(name || 'A').toUpperCase();
  if (!state.rigs[k]) throw new Error(`no camera '${k}' (have ${Object.keys(state.rigs).join(', ')})`);
  state.cam = k;
  if (f0 !== null) state.range.f0 = Math.max(0, Math.min(state.show.n_frames - 1, f0 | 0));
  if (f1 !== null) state.range.f1 = Math.max(state.range.f0 + 1, Math.min(state.show.n_frames, f1 | 0));
  return { camera: k, ...state.range };
}

function seek(i) {
  const n = state.show.n_frames;
  const f = Math.max(0, Math.min(n - 1, i | 0));
  for (const key of Object.keys(state.hosts)) {
    const tr = state.show.tracks[key];
    applyRow(state.hosts[key], tr, tr.values[f]);
  }
  const span = Math.max(1, state.range.f1 - state.range.f0);
  applyRig(camera, state.rigs[state.cam], (f - state.range.f0) / span);
  scene.updateMatrixWorld(true);
  renderer.render(scene, camera);
  state.frame = f;
  return f;
}

function rest() {
  for (const key of Object.keys(state.hosts)) {
    applyRow(state.hosts[key], state.show.tracks[key], restRow(key));
  }
  applyRig(camera, state.rigs[state.cam], 0);
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  const res = await fetch(SHOW_URL);
  if (!res.ok) throw new Error(`show.json not found at ${SHOW_URL} (run scripts/video/show_build.py)`);
  state.show = await res.json();
  for (const key of Object.keys(state.show.tracks)) {
    if (!HOSTS[key]) throw new Error(`show.json has an unknown host '${key}'`);
  }
  for (const [key, name] of Object.entries(HOSTS)) state.hosts[key] = await loadHost(key, name);

  // rest pose first: the eyeline solve measures the head where the show starts it
  for (const key of Object.keys(state.hosts)) {
    applyRow(state.hosts[key], state.show.tracks[key], restRow(key));
  }
  placeHosts();
  const boxes = coveredBoxes();
  const heads = {};
  for (const key of Object.keys(state.hosts)) {
    applyRow(state.hosts[key], state.show.tracks[key], restRow(key));
  }
  scene.updateMatrixWorld(true);
  for (const key of Object.keys(state.hosts)) heads[key] = worldPos(state.hosts[key].head);

  const { w, h } = resize();
  state.boxes = boxes;
  state.rigs = cameraRigs(heads, boxes, w / h, {
    plinthTop: Math.min(...Object.values(state.hosts).map((x) => x.plinthHeight)),
  });
  state.range = { f0: 0, f1: state.show.n_frames };
  setCamera(CAM, params.has('f0') ? parseInt(params.get('f0'), 10) : null,
            params.has('f1') ? parseInt(params.get('f1'), 10) : null);
  seek(params.has('frame') ? parseInt(params.get('frame'), 10) : 0);
  return info();
}

function info() {
  return {
    fps: state.show.fps, n_frames: state.show.n_frames, seconds: state.show.seconds,
    camera: state.cam, range: { ...state.range }, frame: state.frame,
    placeholder_voice: !!state.show.placeholder_voice,
    canvas: { width: renderer.domElement.width, height: renderer.domElement.height },
    lines: state.show.lines.length, sections: state.show.sections.length,
    errors: state.errors.slice(),
  };
}

/**
 * How far each host's gaze is from pointing straight at the other, in degrees.
 * 0 = looking right at them. This is the check that the listener's gaze offset
 * has the right sign for each robot: a listening host's number must be SMALLER
 * than its number at rest, not larger.
 */
function offAxis() {
  const l = state.hosts.lamp, r = state.hosts.reachy;
  if (!l || !r) return null;
  const pair = (a, b) => {
    scene.updateMatrixWorld(true);
    const toOther = worldPos(b.head).sub(worldPos(a.head)).setY(0).normalize();
    const g = gazeDir(a).setY(0).normalize();
    return +THREE.MathUtils.radToDeg(Math.acos(Math.min(1, Math.max(-1, g.dot(toOther))))).toFixed(1);
  };
  return { lamp_to_reachy: pair(l, r), reachy_to_lamp: pair(r, l) };
}

/** Seek, then report the gaze geometry at that frame. */
function measure(frame) {
  seek(frame);
  return { frame: state.frame, offAxisDeg: offAxis() };
}

function debug() {
  const out = { eyeline: EYELINE, hosts: {}, cameras: {} };
  for (const key of Object.keys(state.hosts)) {
    const h = state.hosts[key];
    const b = new THREE.Box3().setFromObject(h.robot);
    out.hosts[key] = {
      urdf: h.profile.description.urdf,
      mount: h.mount.position.toArray().map((v) => +v.toFixed(4)),
      yawDeg: +THREE.MathUtils.radToDeg(h.mount.rotation.y).toFixed(1),
      plinth: +h.plinthHeight.toFixed(4),
      head: worldPos(h.head).toArray().map((v) => +v.toFixed(4)),
      box: { min: b.min.toArray().map((v) => +v.toFixed(4)), max: b.max.toArray().map((v) => +v.toFixed(4)) },
      gaze: gazeDir(h).toArray().map((v) => +v.toFixed(3)),
    };
  }
  out.offAxisDeg = offAxis();
  for (const [k, b] of Object.entries(state.boxes || {})) {
    out.cameras[`box_${k}`] = { min: b.min.toArray().map((v) => +v.toFixed(4)),
                                max: b.max.toArray().map((v) => +v.toFixed(4)) };
  }
  for (const [k, r] of Object.entries(state.rigs || {})) {
    out.cameras[k] = { fov: r.fov, push: r.push || 0,
                       position: r.position.toArray().map((v) => +v.toFixed(4)),
                       target: r.target.toArray().map((v) => +v.toFixed(4)) };
  }
  return out;
}

window.addEventListener('error', (e) => state.errors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => state.errors.push(`promise: ${e.reason}`));
window.addEventListener('resize', () => { resize(); if (state.rigs) seek(Math.max(0, state.frame)); });

window.podcast = {
  seek, setCamera, info, debug, rest, measure,
  /** Batch grab: seek + read the canvas back, one round trip per chunk of frames. */
  grab: (indices, mime = 'image/png') => indices.map((i) => { seek(i); return renderer.domElement.toDataURL(mime); }),
  get show() { return state.show; },
  ready: boot().then((i) => { window.podcast.readyInfo = i; return i; }),
};
window.podcast.ready.catch((e) => {
  state.errors.push(`boot: ${e && e.message ? e.message : e}`);
  const el = document.getElementById('fail');
  el.hidden = false;
  el.textContent = String(e && e.message ? e.message : e);
});
