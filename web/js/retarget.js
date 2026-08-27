// Canonical human motion → robot joints, in the browser.
//
// This is a line-for-line port of `animacy/retarget.py:LiveRetargeter.step`
// and `to_urdf_values` (spec with a numeric example in docs/RETARGET.md). The
// Python side is the reference; if you change one, change the other
// (tests/test_web_retarget_parity.py runs both on the same input and diffs).
//
// Per joint and frame:
//   u = rest + offset + Σ gain·channel
//     → deadband (|x| < db → 0, else x − sign(x)·db)
//     → soft limit (tanh knee over the last `soft_limit` of the range)
//     → clamp [min, max]           (mapping bounds, already resolved by
//                                    `profile.to_web_json`; else joint bounds)
//     → settle toward rest+offset   (after `quiet` s of stillness and no speech; §settle)
//     → + gated idle sway          (only while the mapped target is near-still)
//     → clamp
//     → tracker: spring (exact zero-order-hold) | one-pole, alpha = 1 − exp(−2π·cutoff·dt)
//     → velocity clip               |Δ| ≤ max_speed·dt
//     → clamp [joint.min, joint.max]
//     → carry the spring velocity  (re-derived only if the limits engaged)
//
// Profile JSON is what `animacy profile export` writes (web/robots/<name>.json):
// mapping = {terms[], offset, min, max, deadband, smooth_hz, spring{hz,zeta}|null,
//            idle{amp,hz,still}|null, soft_limit|null}.

// Idle sway generator (docs/RETARGET.md §idle): three sines around `hz`.
export const IDLE_RATIOS = [1.0, 1.31, 0.67];
export const IDLE_WEIGHTS = [0.5, 0.3, 0.2];
export const IDLE_GOLDEN = 2.39996322972865332;
export const IDLE_RELEASE_S = 0.5;

/** Fixed phase of sine k (0..2) for the joint at `jointIndex` in profile.joints. */
export function idlePhase(jointIndex, k) {
  return (IDLE_GOLDEN * (3 * jointIndex + k + 1)) % (2.0 * Math.PI);
}

/** Band-limited sway at time t (seconds on the joint's idle clock). */
export function idleValue(t, amp, hz, jointIndex) {
  let s = 0.0;
  for (let k = 0; k < 3; k++) s += IDLE_WEIGHTS[k] * Math.sin(2.0 * Math.PI * IDLE_RATIOS[k] * hz * t + idlePhase(jointIndex, k));
  return amp * s;
}

/** tanh knee over the last `frac` of the range at each end; identity for frac null/0. */
export function softClip(x, lo, hi, frac) {
  if (!frac) return x;
  const k = frac * (hi - lo);
  if (k <= 0) return x;
  const top = hi - k, bot = lo + k;
  if (x > top) return top + k * Math.tanh((x - top) / k);
  if (x < bot) return bot - k * Math.tanh((bot - x) / k);
  return x;
}

/**
 * Exact zero-order-hold step of y'' = w²(u − y) − 2ζw y' (w = 2π·hz) over dt,
 * as the four coefficients of the linear map [y − u, v] → [y' − u, v']:
 * [pp, pv, vp, vv]. Closed form per damping regime (under / critical / over),
 * so any hz and dt is stable and the overshoot is the analytic exp(−πζ/√(1−ζ²)).
 * Same equations as animacy.retarget.spring_coefficients.
 */
export function springCoefficients(hz, zeta, dt) {
  const w = 2.0 * Math.PI * hz;
  if (Math.abs(zeta - 1.0) < 1e-9) {
    const e = Math.exp(-w * dt);
    const te = dt * e;
    const tef = te * w;
    return [tef + e, te, -w * tef, -tef + e];
  }
  if (zeta < 1.0) {
    const wz = w * zeta;
    const a = w * Math.sqrt(1.0 - zeta * zeta);
    const e = Math.exp(-wz * dt);
    const c = Math.cos(a * dt);
    const s = Math.sin(a * dt);
    const es = e * s, ec = e * c;
    const ewzs = (e * wz * s) / a;
    return [ec + ewzs, es / a, -es * a - wz * ewzs, ec - ewzs];
  }
  const za = -w * zeta;
  const zb = w * Math.sqrt(zeta * zeta - 1.0);
  const z1 = za - zb, z2 = za + zb;
  const e1 = Math.exp(z1 * dt), e2 = Math.exp(z2 * dt);
  const inv = 1.0 / (2.0 * zb);
  const e1i = e1 * inv, e2i = e2 * inv;
  return [e1i * z2 - z2 * e2i + e2, -e1i + e2i, (z1 * e1i - z2 * e2i + e2) * z2, -z1 * e1i + z2 * e2i];
}

/** One exact step of the damped spring toward u (springCoefficients). Returns [y, v]. */
export function springStep(y, v, u, dt, hz, zeta) {
  const [pp, pv, vp, vv] = springCoefficients(hz, zeta, dt);
  const p = y - u;
  return [p * pp + v * pv + u, p * vp + v * vv];
}

/**
 * Steps 8–10 of the per-frame update (animacy.retarget.clip_step): rate limit,
 * hard clamp, and the velocity to carry — the tracker's own vFree when nothing
 * clipped, else the achieved (y − prev)/dt so a spring cannot wind up against a
 * limit. One-pole joints pass vFree = null (velocity unused). Returns [y, v].
 */
export function clipStep(prev, yFree, vFree, dt, maxSpeed, lo, hi) {
  const vmax = maxSpeed * dt;
  const d = yFree - prev;
  let clipped = Math.abs(d) > vmax;
  let y = prev + Math.min(Math.max(d, -vmax), vmax);
  if (y < lo || y > hi) {
    clipped = true;
    y = Math.min(Math.max(y, lo), hi);
  }
  if (vFree === null || vFree === undefined || clipped) return [y, (y - prev) / dt];
  return [y, vFree];
}

const isNil = (v) => v === null || v === undefined;

export class LiveRetargeter {
  /**
   * @param {object} profile   parsed web/robots/<name>.json
   * @param {string} mode      key of profile.retarget (e.g. "default", "puppet")
   * @param {number} defaultSmoothHz  cutoff for joints whose mapping has no smooth_hz
   */
  constructor(profile, mode = 'default', defaultSmoothHz = 6.0) {
    if (!profile.retarget || !profile.retarget[mode]) {
      throw new Error(`robot ${profile.name} has no retarget mode '${mode}'; modes: ${Object.keys(profile.retarget || {})}`);
    }
    this.profile = profile;
    this.mode = mode;
    this.mp = profile.retarget[mode];
    this.defaultSmoothHz = defaultSmoothHz;
    this.reset();
  }

  reset() {
    this.state = {};        // last output
    this.vel = {};          // spring velocity
    this.env = {};          // idle activity envelope
    this.prevTarget = {};   // previous pre-idle target
    this.clock = {};        // idle clock
    this.settlePrev = {};   // settle: previous raw target
    this.quiet = {};        // settle: seconds of quiet so far
    this.blend = {};        // settle: 0 (free) → 1 (at home)
    for (const j of this.profile.joints) {
      this.state[j.name] = j.rest;
      this.vel[j.name] = 0.0;
      this.env[j.name] = 0.0;
      this.prevTarget[j.name] = j.rest;
      this.clock[j.name] = 0.0;
      this.settlePrev[j.name] = j.rest;
      this.quiet[j.name] = 0.0;
      this.blend[j.name] = 0.0;
    }
  }

  /**
   * One frame in, one frame out.
   * @param {Object<string, number>} channels  canonical channels (missing/NaN/null → 0)
   * @param {number} dt  seconds since the previous step
   * @returns {Object<string, number>} joint values in the profile's units (deg / mm)
   */
  step(channels, dt) {
    const out = {};
    const joints = this.profile.joints;
    let speaking = channels.speaking;
    if (speaking === undefined || speaking === null || Number.isNaN(speaking)) speaking = 0;
    for (let idx = 0; idx < joints.length; idx++) {
      const j = joints[idx];
      const m = this.mp[j.name];
      const lo = !m || isNil(m.min) ? j.min : m.min;
      const hi = !m || isNil(m.max) ? j.max : m.max;
      let u, cutoff;
      if (!m) {
        u = j.rest;
        cutoff = this.defaultSmoothHz;
      } else {
        let x = 0.0;
        for (const term of m.terms) {
          let v = channels[term.from];
          if (v === undefined || v === null || Number.isNaN(v)) v = 0.0;
          x += term.gain * Number(v);
        }
        if (m.deadband > 0) x = Math.abs(x) < m.deadband ? 0.0 : x - Math.sign(x) * m.deadband;
        u = x + m.offset + j.rest;
        u = softClip(u, lo, hi, m.soft_limit);
        u = Math.min(Math.max(u, lo), hi);
        cutoff = isNil(m.smooth_hz) ? this.defaultSmoothHz : m.smooth_hz;
        if (m.settle) {
          // step 5b (docs/RETARGET.md §settle == animacy.retarget.settle_update): after `quiet` s of
          // (target speed < still AND not speaking) blend toward home = rest + offset over `seconds`;
          // any motion or speech resets the quiet timer and releases the blend over seconds/4
          const a = Math.abs(u - this.settlePrev[j.name]) / dt;
          this.settlePrev[j.name] = u;
          const quietNow = a < m.settle.still && speaking < 0.5;
          this.quiet[j.name] = quietNow ? this.quiet[j.name] + dt : 0.0;
          const bUp = Math.min(Math.max((this.quiet[j.name] - m.settle.quiet) / m.settle.seconds, 0.0), 1.0);
          const b = Math.max(bUp, this.blend[j.name] - (4.0 * dt) / m.settle.seconds);
          this.blend[j.name] = b;
          u = u + b * (j.rest + m.offset - u);
        }
        if (m.idle) {
          const a = Math.abs(u - this.prevTarget[j.name]) / dt;
          this.prevTarget[j.name] = u;
          const e = Math.max(a, this.env[j.name] * Math.exp(-dt / IDLE_RELEASE_S));
          this.env[j.name] = e;
          const still = isNil(m.idle.still) ? 10.0 * m.idle.amp * m.idle.hz : m.idle.still;
          const g = Math.min(Math.max(1.0 - e / still, 0.0), 1.0);
          this.clock[j.name] += dt;
          u = Math.min(Math.max(u + g * idleValue(this.clock[j.name], m.idle.amp, m.idle.hz, idx), lo), hi);
        }
      }
      const prev = this.state[j.name];
      let yFree, vFree;
      if (m && m.spring) {
        [yFree, vFree] = springStep(prev, this.vel[j.name], u, dt, m.spring.hz, m.spring.zeta);
      } else {
        // one-pole low-pass, alpha from cutoff
        const alpha = !cutoff ? 1.0 : 1.0 - Math.exp(-2.0 * Math.PI * cutoff * dt);
        yFree = prev + alpha * (u - prev);
        vFree = null;
      }
      const [y, v] = clipStep(prev, yFree, vFree, dt, j.max_speed, j.min, j.max);
      this.vel[j.name] = v;
      this.state[j.name] = y;
      out[j.name] = y;
    }
    return out;
  }
}

/**
 * Joint values (profile units) → URDF joint values (radians / metres).
 * Mirrors `animacy.retarget.to_urdf_values`: (value + urdf_offset) * urdf_sign,
 * then deg→rad or mm→m. Keys are `urdf_joint` names.
 *
 * @param {Object<string, number>} values  joint name → value in profile units
 * @param {object} profile
 * @param {object} [alias]  optional dev-only remap {jointName: {urdf_joint, sign, offset}}
 *                          applied in profile units BEFORE the profile's own sign/offset
 *                          (used for stand-in URDFs whose joint names/zeros differ)
 */
export function toUrdfValues(values, profile, alias = null) {
  const out = {};
  for (const j of profile.joints) {
    if (!(j.name in values)) continue;
    let v = values[j.name];
    let urdfJoint = j.urdf_joint || j.name;
    if (alias && alias[j.name]) {
      const a = alias[j.name];
      v = (v + (a.offset || 0)) * (a.sign === undefined ? 1 : a.sign);
      urdfJoint = a.urdf_joint || urdfJoint;
    }
    v = (v + j.urdf_offset) * j.urdf_sign;
    if (j.unit === 'deg') v = (v * Math.PI) / 180.0;
    else if (j.unit === 'mm') v = v / 1000.0;
    out[urdfJoint] = v;
  }
  return out;
}

/** Rest pose of a profile, in profile units. */
export function restValues(profile) {
  const out = {};
  for (const j of profile.joints) out[j.name] = j.rest;
  return out;
}
