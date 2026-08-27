// The canonical human motion space (`animacy.human.v1`) in the browser, and
// the derivation of every channel from MediaPipe's FaceLandmarker /
// PoseLandmarker output.
//
// docs/CANONICAL.md is the contract. The Python capture module
// (animacy/capture.py) is written from the same document; the SIGNS below are
// what both must agree on, so every one is derived explicitly in comments.
//
// Frame (ROS body frame, viewed from the subject):
//     x forward (toward the camera)   y left (subject's left)   z up
//
// ---------------------------------------------------------------------------
// Channel table (mirrors animacy/schema.py `_SPEC`: name, unit, lo, hi)
// ---------------------------------------------------------------------------
const SPEC = [
  ['t', 's', 0, Infinity],
  ['head_yaw', 'deg', -90, 90],
  ['head_pitch', 'deg', -60, 60],
  ['head_roll', 'deg', -45, 45],
  ['head_x', 'mm', -150, 150],
  ['head_y', 'mm', -150, 150],
  ['head_z', 'mm', -150, 150],
  ['gaze_yaw', 'deg', -40, 40],
  ['gaze_pitch', 'deg', -30, 30],
  ['brow_l', 'unit', 0, 1],
  ['brow_r', 'unit', 0, 1],
  ['brow_furrow', 'unit', 0, 1],
  ['eye_open_l', 'unit', 0, 1],
  ['eye_open_r', 'unit', 0, 1],
  ['mouth_open', 'unit', 0, 1],
  ['smile', 'unit', 0, 1],
  ['torso_lean_fwd', 'deg', -45, 45],
  ['torso_lean_side', 'deg', -45, 45],
  ['torso_yaw', 'deg', -90, 90],
  ['arm_valid', 'flag', 0, 1],
  ['shoulder_yaw', 'deg', -90, 90],
  ['shoulder_pitch', 'deg', -30, 180],
  ['elbow_flex', 'deg', 0, 150],
  ['wrist_roll', 'deg', -90, 90],
  ['wrist_pitch', 'deg', -80, 80],
  ['hand_open', 'unit', 0, 1],
  ['speaking', 'flag', 0, 1],
  ['face_valid', 'flag', 0, 1],
];

export const CHANNELS = SPEC.map((s) => s[0]);
export const UNITS = Object.fromEntries(SPEC.map((s) => [s[0], s[1]]));
export const BOUNDS = Object.fromEntries(SPEC.map((s) => [s[0], [s[2], s[3]]]));
export const FLAGS = CHANNELS.filter((c) => UNITS[c] === 'flag');
export const MAPPABLE = CHANNELS.filter((c) => c !== 't' && UNITS[c] !== 'flag');
export const FACE_CHANNELS = [
  'head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z',
  'gaze_yaw', 'gaze_pitch', 'brow_l', 'brow_r', 'brow_furrow',
  'eye_open_l', 'eye_open_r', 'mouth_open', 'smile',
];
export const ARM_CHANNELS = ['shoulder_yaw', 'shoulder_pitch', 'elbow_flex', 'wrist_roll', 'wrist_pitch', 'hand_open'];
export const TORSO_CHANNELS = ['torso_lean_fwd', 'torso_lean_side', 'torso_yaw'];

// Channels that are zeroed against the neutral pose (angles, translations and
// the affect sliders whose resting value is not 0). eye_open_* stays absolute:
// its neutral is ≈0.6 by definition (schema.empty_frames).
const NEUTRAL_RELATIVE = [
  'head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z',
  'gaze_yaw', 'gaze_pitch', 'brow_l', 'brow_r', 'brow_furrow', 'mouth_open', 'smile',
  'torso_lean_fwd', 'torso_lean_side', 'torso_yaw',
];
const UNIT_CLAMP01 = ['brow_l', 'brow_r', 'brow_furrow', 'mouth_open', 'smile', 'eye_open_l', 'eye_open_r', 'hand_open'];

/** A neutral frame: zeros everywhere, eyes half open, flags 0. */
export function neutralFrame() {
  const f = {};
  for (const c of CHANNELS) f[c] = 0.0;
  f.eye_open_l = 0.6;
  f.eye_open_r = 0.6;
  return f;
}

const RAD = 180 / Math.PI;
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const v3 = (x, y, z) => ({ x, y, z });
const sub = (a, b) => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const add = (a, b) => v3(a.x + b.x, a.y + b.y, a.z + b.z);
const scale = (a, s) => v3(a.x * s, a.y * s, a.z * s);
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const norm = (a) => Math.hypot(a.x, a.y, a.z);
const unit = (a) => { const n = norm(a) || 1e-9; return scale(a, 1 / n); };
const mid = (a, b) => scale(add(a, b), 0.5);

// ---------------------------------------------------------------------------
// FaceLandmarker → raw face channels
// ---------------------------------------------------------------------------
//
// MediaPipe's `facialTransformationMatrixes[0]` is the pose of the canonical
// face model in the camera's *metric 3D space* (translation in cm):
//   camera space: right-handed, origin at the camera, camera looks down −Z.
//   +X = image right, +Y = up, +Z = toward the viewer (out of the screen).
// The raw getUserMedia frame is NOT mirrored, so "image right" is the
// subject's LEFT, and "toward the viewer" is the subject's FORWARD.
//
// Canonical face model axes (identity when the subject looks straight at the
// camera): +x = subject's left, +y = up, +z = forward (toward the camera).
// That is exactly the camera frame's orientation, so R is the head rotation.
//
// Camera → body frame (x fwd, y left, z up) is the axis permutation
//   body.x = cam.z,  body.y = cam.x,  body.z = cam.y      (det +1, proper)
// Then ZYX Euler (yaw about z, pitch about y, roll about x):
//   yaw   = atan2(R10, R00)   + = forward axis swings toward +y (LEFT)     ✓ CANONICAL
//   pitch = asin(−R20)        ROS +pitch about +y tips the nose DOWN;
//                             CANONICAL says + = UP  →  head_pitch = −pitch
//   roll  = atan2(R21, R22)   + about +x (forward) lifts the LEFT ear, i.e.
//                             the RIGHT ear drops toward the right shoulder ✓
// Translation: head_x = +Δcam.z (lean toward the camera), head_y = +Δcam.x
// (subject's left), head_z = +Δcam.y (up); cm → mm (×10).
//
// Blendshapes are ARKit-named and subject-relative ("Left" = subject's left).

function readMatrix4(data) {
  // MediaPipe reports a 4x4; the storage order differs between builds. The
  // translation is tens of cm while rotation entries are ≤ 1, so detect it.
  const colMajor = Math.abs(data[14]) > Math.abs(data[11]);
  const at = colMajor ? (r, c) => data[c * 4 + r] : (r, c) => data[r * 4 + c];
  const R = [[at(0, 0), at(0, 1), at(0, 2)], [at(1, 0), at(1, 1), at(1, 2)], [at(2, 0), at(2, 1), at(2, 2)]];
  const t = [at(0, 3), at(1, 3), at(2, 3)];
  return { R, t };
}

function blend(map, name) {
  const v = map[name];
  return v === undefined ? 0 : v;
}

/**
 * @param {object} res  FaceLandmarker.detectForVideo result
 * @returns {object|null} raw (absolute, pre-neutral) face channels + face_valid
 */
export function faceToRaw(res) {
  if (!res || !res.faceLandmarks || res.faceLandmarks.length === 0) return null;
  const out = {};
  const M = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
  if (M && M.data && M.data.length >= 16) {
    const { R: Rc, t } = readMatrix4(M.data);
    // permute camera → body: body index b takes camera index P[b]
    const P = [2, 0, 1]; // body.x=cam.z, body.y=cam.x, body.z=cam.y
    const R = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Rc[P[i]][P[j]]));
    const yaw = Math.atan2(R[1][0], R[0][0]) * RAD;
    const pitch = Math.asin(clamp(-R[2][0], -1, 1)) * RAD;
    const roll = Math.atan2(R[2][1], R[2][2]) * RAD;
    out.head_yaw = yaw;          // + = subject's left
    out.head_pitch = -pitch;     // + = up
    out.head_roll = roll;        // + = right ear drops
    out.head_x = t[2] * 10.0;    // cm → mm, + toward the camera
    out.head_y = t[0] * 10.0;    // + subject's left
    out.head_z = t[1] * 10.0;    // + up
  } else {
    // No matrix → head pose unknown; leave NaN so consumers mask it.
    for (const c of ['head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z']) out[c] = NaN;
  }
  const bs = {};
  const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
  if (cats) for (const c of cats) bs[c.categoryName] = c.score;
  // brows: raise = outer/inner up; furrow = brows down/in
  out.brow_l = Math.max(blend(bs, 'browOuterUpLeft'), blend(bs, 'browInnerUp'));
  out.brow_r = Math.max(blend(bs, 'browOuterUpRight'), blend(bs, 'browInnerUp'));
  out.brow_furrow = 0.5 * (blend(bs, 'browDownLeft') + blend(bs, 'browDownRight'));
  // eyes: 0 closed … 1 wide open
  out.eye_open_l = 1.0 - blend(bs, 'eyeBlinkLeft');
  out.eye_open_r = 1.0 - blend(bs, 'eyeBlinkRight');
  out.mouth_open = blend(bs, 'jawOpen');
  out.smile = 0.5 * (blend(bs, 'mouthSmileLeft') + blend(bs, 'mouthSmileRight'));
  // gaze relative to the head, + = left / up. For the LEFT eye "out" is left;
  // for the RIGHT eye "in" is left. Blendshape 1.0 ≈ 35° yaw / 25° pitch.
  const gy = 0.5 * ((blend(bs, 'eyeLookOutLeft') - blend(bs, 'eyeLookInLeft')) +
                    (blend(bs, 'eyeLookInRight') - blend(bs, 'eyeLookOutRight')));
  const gp = 0.5 * ((blend(bs, 'eyeLookUpLeft') - blend(bs, 'eyeLookDownLeft')) +
                    (blend(bs, 'eyeLookUpRight') - blend(bs, 'eyeLookDownRight')));
  out.gaze_yaw = 35.0 * gy;
  out.gaze_pitch = 25.0 * gp;
  out.face_valid = 1;
  return out;
}

// ---------------------------------------------------------------------------
// PoseLandmarker → raw torso + puppet-arm channels
// ---------------------------------------------------------------------------
//
// `worldLandmarks` are metres, origin at the hip centre, axes like the image:
//   +X = image right (subject's LEFT), +Y = DOWN, +Z = away from the camera.
// Camera → body:  body.x = −Z (toward camera), body.y = +X (left), body.z = −Y (up)
// (det +1). Landmark indices: 11/12 L/R shoulder, 13/14 elbow, 15/16 wrist,
// 17/18 pinky, 19/20 index, 21/22 thumb, 23/24 hip. "Left" is the subject's.
//
// torso: u = shoulder_centre − hip_centre (the spine, ≈ +z at neutral)
//   torso_lean_fwd  = atan2(u.x, u.z)   + = top of the spine toward the camera
//   torso_lean_side = atan2(u.y, u.z)   + = toward the subject's left
//   d = L_shoulder − R_shoulder (≈ +y at neutral); a left turn (yaw + about z)
//   rotates +y toward −x:  torso_yaw = atan2(−d.x, d.y)
//
// puppet arm (right arm by default; `arm: 'left'` mirrors y so downstream
// never cares):  a = elbow − shoulder, f = wrist − elbow, h = index − wrist
//   shoulder_pitch = atan2(a.x, −a.z)   0 hanging, 90 forward, 180 up
//   shoulder_yaw   = atan2(a.y, a.x)    + = swings left (azimuth about z)
//   elbow_flex     = 180 − ∠(−a, f)     0 straight, + bending
//   wrist_pitch    = elevation of h above the forearm axis, + = hand tips up
//   wrist_roll     = angle of (thumb − pinky) about the forearm axis, 0 = thumb
//                    up, + = thumb rotates toward the subject's left (pronation)
//   hand_open      = index↔pinky spread / forearm length, scaled to 0..1
const POSE = { LS: 11, RS: 12, LE: 13, RE: 14, LW: 15, RW: 16, LP: 17, RP: 18, LI: 19, RI: 20, LT: 21, RT: 22, LH: 23, RH: 24 };

function toBody(p, mirror) {
  const b = v3(-p.z, p.x, -p.y);
  if (mirror) b.y = -b.y;
  return b;
}

/**
 * @param {object} res  PoseLandmarker.detectForVideo result
 * @param {'right'|'left'} arm  which arm is the puppet arm
 * @returns {object|null} raw torso + arm channels + arm_valid
 */
export function poseToRaw(res, arm = 'right') {
  if (!res || !res.worldLandmarks || res.worldLandmarks.length === 0) return null;
  const W = res.worldLandmarks[0];
  const L = (res.landmarks && res.landmarks[0]) || W;
  const vis = (i) => {
    const v = L[i] && L[i].visibility;
    return v === undefined || v === null ? 1.0 : v;
  };
  const P = (i) => toBody(W[i], false);
  const out = {};
  // torso
  const S = mid(P(POSE.LS), P(POSE.RS));
  const H = mid(P(POSE.LH), P(POSE.RH));
  const torsoOk = Math.min(vis(POSE.LS), vis(POSE.RS)) > 0.5;
  if (torsoOk) {
    const u = sub(S, H);
    const uz = Math.max(u.z, 1e-3);
    out.torso_lean_fwd = Math.atan2(u.x, uz) * RAD;
    out.torso_lean_side = Math.atan2(u.y, uz) * RAD;
    const d = sub(P(POSE.LS), P(POSE.RS));
    out.torso_yaw = Math.atan2(-d.x, d.y) * RAD;
  } else {
    out.torso_lean_fwd = NaN; out.torso_lean_side = NaN; out.torso_yaw = NaN;
  }
  // puppet arm
  const mirror = arm === 'left';
  const idx = mirror
    ? { S: POSE.LS, E: POSE.LE, Wr: POSE.LW, I: POSE.LI, Pk: POSE.LP, T: POSE.LT }
    : { S: POSE.RS, E: POSE.RE, Wr: POSE.RW, I: POSE.RI, Pk: POSE.RP, T: POSE.RT };
  const armOk = arm !== 'none' && Math.min(vis(idx.S), vis(idx.E), vis(idx.Wr)) > 0.5;
  out.arm_valid = armOk ? 1 : 0;
  if (armOk) {
    const B = (i) => toBody(W[i], mirror);
    const sh = B(idx.S), el = B(idx.E), wr = B(idx.Wr), ix = B(idx.I), pk = B(idx.Pk), th = B(idx.T);
    const a = sub(el, sh);
    const f = sub(wr, el);
    const h = sub(ix, wr);
    out.shoulder_pitch = Math.atan2(a.x, -a.z) * RAD;
    out.shoulder_yaw = Math.atan2(a.y, a.x) * RAD;
    const ua = unit(scale(a, -1)), uf = unit(f);
    out.elbow_flex = 180 - Math.acos(clamp(dot(ua, uf), -1, 1)) * RAD;
    // forearm-relative frame: fwd = f̂, up = z ⟂ f̂, left = up × fwd
    const fwd = uf;
    const zAxis = v3(0, 0, 1);
    let up = sub(zAxis, scale(fwd, dot(zAxis, fwd)));
    if (norm(up) < 1e-3) up = v3(1, 0, 0);
    up = unit(up);
    const left = unit(cross(up, fwd));
    out.wrist_pitch = Math.atan2(dot(h, up), dot(h, fwd)) * RAD;
    const l = sub(th, pk);
    out.wrist_roll = Math.atan2(dot(l, left), dot(l, up)) * RAD;
    const spread = norm(sub(ix, pk)) / (norm(f) || 1e-6);
    out.hand_open = clamp((spread - 0.15) / 0.25, 0, 1);
  } else {
    for (const c of ARM_CHANNELS) out[c] = NaN;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Neutral calibration: median of the last ~1 s of raw frames
// ---------------------------------------------------------------------------
export class NeutralCalibrator {
  constructor(windowS = 1.0) {
    this.windowS = windowS;
    this.buf = [];        // [{t, raw}]
    this.neutral = null;  // channel → median raw value
  }

  push(raw, t) {
    this.buf.push({ t, raw });
    while (this.buf.length && t - this.buf[0].t > this.windowS) this.buf.shift();
  }

  /** Freeze the median of the buffer as the neutral pose. Returns it (or null if empty). */
  setNeutral() {
    if (!this.buf.length) return null;
    const n = {};
    for (const c of NEUTRAL_RELATIVE) {
      const vals = this.buf.map((s) => s.raw[c]).filter((v) => v !== undefined && !Number.isNaN(v)).sort((a, b) => a - b);
      if (!vals.length) continue;
      const m = vals.length >> 1;
      n[c] = vals.length % 2 ? vals[m] : 0.5 * (vals[m - 1] + vals[m]);
    }
    this.neutral = n;
    return n;
  }

  get hasNeutral() { return this.neutral !== null; }
  get bufferSeconds() { return this.buf.length ? this.buf[this.buf.length - 1].t - this.buf[0].t : 0; }
}

/**
 * Raw (absolute) channels → canonical frame relative to `neutral`.
 * Channels absent from `raw` stay NaN unless they are flags (0).
 * @param {object} raw     merged faceToRaw + poseToRaw
 * @param {object|null} neutral  from NeutralCalibrator.setNeutral (null → raw as-is)
 * @param {number} t
 * @param {number} speaking  0/1 from the VAD
 */
export function toCanonical(raw, neutral, t = 0, speaking = 0) {
  const f = {};
  for (const c of CHANNELS) f[c] = FLAGS.includes(c) ? 0 : NaN;
  f.t = t;
  for (const c of CHANNELS) {
    if (c === 't') continue;
    let v = raw[c];
    if (v === undefined || v === null) continue;
    if (neutral && NEUTRAL_RELATIVE.includes(c) && neutral[c] !== undefined) v -= neutral[c];
    if (UNIT_CLAMP01.includes(c) && !Number.isNaN(v)) v = clamp(v, 0, 1);
    f[c] = v;
  }
  f.speaking = speaking ? 1 : 0;
  f.face_valid = raw.face_valid ? 1 : 0;
  f.arm_valid = raw.arm_valid ? 1 : 0;
  return f;
}
