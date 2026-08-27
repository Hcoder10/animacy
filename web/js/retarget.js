// Canonical human motion → robot joints, in the browser.
//
// This is a line-for-line port of `animacy/retarget.py:LiveRetargeter.step`
// and `to_urdf_values`. The Python side is the reference; if you change one,
// change the other (tests/test_web_retarget_parity.py runs both on the same
// input and diffs the output).
//
//   joint = rest + offset + Σ gain·channel
//         → deadband (|x| < db → 0, else x − sign(x)·db)
//         → clamp [min, max]           (mapping bounds, already resolved by
//                                        `profile.to_web_json`; else joint bounds)
//         → one-pole low-pass           alpha = 1 − exp(−2π·cutoff·dt)
//         → velocity clip               |Δ| ≤ max_speed·dt
//         → clamp [joint.min, joint.max]
//
// Profile JSON is what `animacy profile export` writes (web/robots/<name>.json).

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
    this.state = {};
    this.reset();
  }

  reset() {
    for (const j of this.profile.joints) this.state[j.name] = j.rest;
  }

  /**
   * One frame in, one frame out.
   * @param {Object<string, number>} channels  canonical channels (missing/NaN/null → 0)
   * @param {number} dt  seconds since the previous step
   * @returns {Object<string, number>} joint values in the profile's units (deg / mm)
   */
  step(channels, dt) {
    const out = {};
    for (const j of this.profile.joints) {
      const m = this.mp[j.name];
      let target, cutoff;
      if (!m) {
        target = j.rest;
        cutoff = this.defaultSmoothHz;
      } else {
        let x = 0.0;
        for (const term of m.terms) {
          let v = channels[term.from];
          if (v === undefined || v === null || Number.isNaN(v)) v = 0.0;
          x += term.gain * Number(v);
        }
        if (m.deadband > 0) {
          x = Math.abs(x) < m.deadband ? 0.0 : x - Math.sign(x) * m.deadband;
        }
        target = x + m.offset + j.rest;
        const lo = m.min === null || m.min === undefined ? j.min : m.min;
        const hi = m.max === null || m.max === undefined ? j.max : m.max;
        target = Math.min(Math.max(target, lo), hi);
        cutoff = m.smooth_hz === null || m.smooth_hz === undefined ? this.defaultSmoothHz : m.smooth_hz;
      }
      const prev = this.state[j.name];
      // one-pole low-pass, alpha from cutoff
      const alpha = !cutoff ? 1.0 : 1.0 - Math.exp(-2.0 * Math.PI * cutoff * dt);
      let y = prev + alpha * (target - prev);
      // velocity clip
      const vmax = j.max_speed * dt;
      y = prev + Math.min(Math.max(y - prev, -vmax), vmax);
      y = Math.min(Math.max(y, j.min), j.max);
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
