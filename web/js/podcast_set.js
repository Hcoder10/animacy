// The podcast set: a dark studio, two plinths, a four-light rig, five cameras.
//
// Nothing here touches motion. The set is built around whatever the two URDFs
// actually measure: the plinth heights are solved so both heads land on one
// eyeline, and every camera is fitted to a box that covers the robots' whole
// range of motion in the show, so nothing can drift out of frame.
//
// World convention is the viewer's: the robot's forward (+x in URDF) is +x in
// three.js, up is +y. The audience is at +x, so from camera A a host at +z is
// on the LEFT of frame and a host at -z is on the RIGHT.

import * as THREE from 'three';

// --- the set ---------------------------------------------------------------
export const EYELINE = 0.56;          // m: where both heads sit
// Reachy's head link sits low in a tall shell, so matching head-link heights put
// its face visibly above the lamp's in the wide. Dropping it a further 2 cm
// levels what the eye actually reads as the two eyelines.
export const REACHY_DROP = 0.035;
// Hosts sit on a line oblique to the camera, not side by side: the lamp is
// nearer and camera-left, the reachy further and camera-right. That is what
// makes an over-the-shoulder possible at all, and it gives the wide some depth.
//
// `gazeAz` is where the host's *gaze* points, in degrees around +y from the
// audience direction (+x), positive turning toward -z. It is not a body yaw:
// a lamp's shade does not point along its URDF's +x, so podcast.js measures the
// rest gaze and solves the body yaw that lands it here. ~34 deg off the lens is
// the classic two-hander: addressing each other, still open to camera. Less than
// about 30 stops reading as "turned toward you" on a shade this round.
export const PLACE = {
  lamp: { x: 0.085, z: 0.235, gazeAz: 34 },
  reachy: { x: -0.085, z: -0.215, gazeAz: -34 },
};
export const PLINTH_R = { lamp: 0.155, reachy: 0.125 };

const COL = {
  bg: 0x05060a,
  floor: 0x0e1117,
  plinth: 0x161923,
  plinthTop: 0x1e2230,
  backdrop: 0x131722,
  key: 0xffd6ac,        // warm
  rim: 0x9dc0ff,        // cool
  bounce: 0xffe6c8,
  wash: 0x5f7bb8,
};

export function buildSet(scene) {
  scene.background = new THREE.Color(COL.bg);
  scene.fog = new THREE.FogExp2(COL.bg, 0.26);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(6, 96),
    new THREE.MeshStandardMaterial({ color: COL.floor, roughness: 0.78, metalness: 0.04 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // A cyclorama behind the hosts, close enough to catch light: the frame gets a
  // floor-to-wall gradient and the rim has something to die into, instead of a
  // flat black card behind everything.
  const cyc = new THREE.Mesh(
    new THREE.CylinderGeometry(2.4, 2.4, 3.0, 64, 1, true),
    new THREE.MeshStandardMaterial({ color: COL.backdrop, roughness: 1.0, metalness: 0.0, side: THREE.BackSide }),
  );
  cyc.position.y = 0.9;
  cyc.receiveShadow = true;
  scene.add(cyc);

  // ambient floor: enough that the shadow sides read as shadow rather than
  // crushed black, not enough to flatten the key
  scene.add(new THREE.HemisphereLight(0x9fb2d8, 0x0a0b0f, 0.24));
  scene.add(new THREE.AmbientLight(0xffffff, 0.085));

  // key: warm, camera-left (+z), in front (+x) and above. The only shadow caster.
  // Kept low: both shells are near-white and clip to paper at anything brighter.
  // 10.5, not more: the lamp's base carries a fan grille and vent slots whose
  // upward faces sit straight under the key. Any brighter and they clip to flat
  // white and read as shattered geometry instead of as detail.
  const key = new THREE.SpotLight(COL.key, 10.5, 0, THREE.MathUtils.degToRad(36), 0.9, 1.6);
  key.position.set(1.02, 1.30, 1.00);
  key.target.position.set(0, EYELINE - 0.08, 0.02);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.4;
  key.shadow.camera.far = 5.0;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.025;
  key.shadow.radius = 4.0;
  scene.add(key, key.target);

  // rim: cool, from behind and camera-right (-x, -z), grazing the shells
  const rim = new THREE.SpotLight(COL.rim, 11.0, 0, THREE.MathUtils.degToRad(44), 0.92, 1.5);
  rim.position.set(-1.10, 1.02, -1.05);
  rim.target.position.set(0.05, EYELINE - 0.02, 0);
  scene.add(rim, rim.target);

  // fill: a low, soft, neutral source from camera-right — the side the key does
  // not reach — so the shadow half of each robot keeps its shape. No shadows of
  // its own; a second shadow caster on a two-hander reads as a mistake.
  const fill = new THREE.SpotLight(0xdfe6f5, 3.6, 0, THREE.MathUtils.degToRad(52), 1.0, 1.4);
  fill.position.set(0.95, 0.72, -0.90);
  fill.target.position.set(0, EYELINE - 0.05, 0);
  scene.add(fill, fill.target);

  // bounce: a soft warm lift from low camera-left, so the plinths are not holes
  const bounce = new THREE.PointLight(COL.bounce, 1.15, 2.8, 2.0);
  bounce.position.set(0.80, 0.20, 0.60);
  scene.add(bounce);

  // a pool on the cyclorama behind the hosts: separation, and no visible source.
  // Placed past them so it cannot spill onto either robot.
  const wash = new THREE.SpotLight(COL.wash, 14.0, 0, THREE.MathUtils.degToRad(50), 1.0, 1.3);
  wash.position.set(-1.30, 1.25, 0.30);
  wash.target.position.set(-2.35, 0.35, -0.15);
  scene.add(wash, wash.target);

  return { floor, cyc, key, rim, fill, bounce, wash };
}

/** A plinth from the floor to `height`, at (x, z). Slight taper reads as a stool. */
export function buildPlinth(scene, x, z, height, radius) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius * 1.06, height, 56),
    new THREE.MeshStandardMaterial({ color: COL.plinth, roughness: 0.72, metalness: 0.06 }),
  );
  body.position.y = height / 2;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);
  const lip = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 1.04, radius * 1.04, 0.012, 56),
    new THREE.MeshStandardMaterial({ color: COL.plinthTop, roughness: 0.5, metalness: 0.12 }),
  );
  lip.position.y = height - 0.006;
  lip.castShadow = true;
  lip.receiveShadow = true;
  g.add(lip);
  g.position.set(x, 0, z);
  scene.add(g);
  return g;
}

// --- cameras ---------------------------------------------------------------
// Every camera is a static tripod: a position and a target, both derived from
// the measured head positions. No orbiting, no bobbing, no handheld noise.
// `push` is the only motion in the set: camera E loses `push` degrees of field
// of view across the take, which reads as a very slow dolly in.

const D2R = Math.PI / 180;

/** Unit vector at azimuth `az` degrees from world +x, rotating +x toward -z (the viewer's yaw sense). */
function dirAt(az, elevation = 0) {
  const a = az * D2R;
  const e = elevation * D2R;
  return new THREE.Vector3(Math.cos(a) * Math.cos(e), Math.sin(e), -Math.sin(a) * Math.cos(e));
}

/** Distance at which a `w` x `h` window fills a `fovDeg` camera of this aspect. */
function distanceFor(w, h, fovDeg, aspect) {
  const tanV = Math.tan((fovDeg * D2R) / 2);
  return Math.max((0.5 * h) / tanV, (0.5 * w) / (tanV * aspect), 0.2);
}

/**
 * Slide the subject sideways in frame without moving the camera: shift the aim
 * point along the camera's own right vector by `frac` of the frame width. The
 * subject travels the other way, so a positive `frac` puts it left of centre —
 * which is where a host looking to frame-right belongs, with the room it is
 * looking into left open. Composition, not a camera move.
 */
function lead(position, target, fovDeg, aspect, frac) {
  if (!frac) return target;
  const fwd = target.clone().sub(position).normalize();
  const right = fwd.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
  const width = 2 * position.distanceTo(target) * Math.tan((fovDeg * D2R) / 2) * aspect;
  return target.clone().add(right.multiplyScalar(frac * width));
}

/**
 * The five cameras.
 *
 * Everything is solved against `boxes` — the union of each robot's bounding box
 * over *every frame of the show*, not its rest pose. Framing against the range
 * of motion is what guarantees a raised shade or a swung antenna cannot leave
 * the frame halfway through a take.
 *
 * @param {{lamp: THREE.Vector3, reachy: THREE.Vector3}} heads world head positions
 * @param {{all: THREE.Box3, lamp: THREE.Box3, reachy: THREE.Box3}} boxes
 * @param {number} aspect
 * @param {{plinthTop: number}} set  the lower of the two plinth tops
 */
export function cameraRigs(heads, boxes, aspect, set) {
  const covered = boxes.all;
  const rigs = {};

  // --- A: the wide two-shot ------------------------------------------------
  // Vertical: from a hand's width of plinth below the hosts to a little air
  // above the tallest thing either of them does in the whole show. Horizontal:
  // the span they are spread across, with slack. The taller requirement wins,
  // so neither axis is ever cropped.
  const bottom = set.plinthTop - 0.13;
  const top = covered.max.y + 0.055;
  const wideH = top - bottom;
  const wideW = Math.max(covered.max.z - covered.min.z, covered.max.x - covered.min.x) * 1.10;
  const wideFov = 30;
  const wideDist = distanceFor(wideW, wideH, wideFov, aspect);
  const wideTarget = new THREE.Vector3(covered.getCenter(new THREE.Vector3()).x,
                                       0.5 * (bottom + top),
                                       0.5 * (covered.min.z + covered.max.z));
  const widePos = wideTarget.clone().add(dirAt(-3, 4).multiplyScalar(wideDist));
  rigs.A = { fov: wideFov, position: widePos, target: wideTarget, push: 0 };
  rigs.E = { fov: wideFov, position: widePos.clone(), target: wideTarget.clone(), push: 1.0 };

  // --- B / C: the singles --------------------------------------------------
  // One host fills the frame. Shot from across the set (from the other host's
  // side of the lens axis) so the robot is seen three-quarter front, and held on
  // its own eyeline: these are characters, not exhibits on a turntable.
  //
  // A wider lens closer in, not a longer one further back: the two hosts are
  // only ~45 cm apart, so on a 30 deg lens a "single" still frames both at the
  // same size and reads as a second wide. Shortening the lens shrinks whoever is
  // behind and leaves the subject alone in frame.
  const single = (box, azimuth, elev, margin, leadFrac) => {
    const c = box.getCenter(new THREE.Vector3());
    const h = (box.max.y - box.min.y) * margin;
    const w = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) * margin;
    const fov = 36;
    const d = distanceFor(w, h, fov, aspect);
    const position = c.clone().add(dirAt(azimuth, elev).multiplyScalar(d));
    return { fov, position, target: lead(position, c, fov, aspect, leadFrac), push: 0 };
  };
  // The lamp turns to frame-right here, so it sits left of centre; the reachy
  // turns to frame-left, so it sits right of centre.
  rigs.B = single(boxes.lamp, 14, 5, 1.14, 0.05);
  rigs.C = single(boxes.reachy, -14, 5, 1.14, -0.06);

  // --- D: over the lamp's shoulder onto the reachy -------------------------
  // The camera sits behind the lamp on the line between the hosts and steps off
  // it toward the audience, so the lamp's shade holds the near edge of frame and
  // the reachy — the subject — sits clear of it, off centre.
  const along = heads.lamp.clone().sub(heads.reachy).setY(0).normalize();   // reachy -> lamp
  const side = new THREE.Vector3(-along.z, 0, along.x);                     // away from the audience
  // The lateral step off the line has to clear the shade: nearly on the line,
  // the lamp's own head covers the subject's face.
  const otsPos = heads.lamp.clone()
    .add(along.clone().multiplyScalar(0.60))
    .add(side.clone().multiplyScalar(-0.42))
    .add(new THREE.Vector3(0, 0.200, 0));
  const subject = boxes.reachy;
  const sc = subject.getCenter(new THREE.Vector3());
  // the subject takes a bit over half the frame height; the rest is the lamp's
  // shoulder and the room
  const otsDist = otsPos.distanceTo(sc);
  const otsFov = Math.min(46, Math.max(22,
    (2 * Math.atan((0.5 * (subject.max.y - subject.min.y)) / 0.60 / otsDist)) / D2R));
  rigs.D = { fov: otsFov, position: otsPos, push: 0,
             target: lead(otsPos, sc, otsFov, aspect, 0.04) };

  return rigs;
}

/** Point `camera` at a rig. `u` in [0,1] is the take's progress (only camera E uses it). */
export function applyRig(camera, rig, u = 0) {
  camera.fov = rig.fov - (rig.push || 0) * Math.min(1, Math.max(0, u));
  camera.position.copy(rig.position);
  camera.up.set(0, 1, 0);
  camera.lookAt(rig.target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
}
