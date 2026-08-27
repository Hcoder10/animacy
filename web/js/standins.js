// DEV-ONLY stand-in robot descriptions.
//
// The shipped viewer loads `robots/<name>/<description.urdf>` from each
// robot's profile. While those URDFs are being authored, the loader falls back
// to the entries below (and shows a "stand-in" badge in the viewport). Pass
// `?standin=1` to force them, `?urdf_<robot>=<url>` to point at any URDF.
//
// The LeLamp simulation URDF is GPL-3.0 *reference* material
// (third_party/README.md) — it is never the shipped default and is only
// reachable here when robots/lamp/urdf/lamp.urdf is missing.

export const STANDINS = {
  lamp: {
    urdf: '../third_party/lelamp/simulation/robot.urdf',
    packages: { assets: '../third_party/lelamp/simulation/assets' },
    note: 'dev stand-in: LeLamp simulation URDF (GPL-3.0 reference, not shipped)',
    // That URDF is rooted at the lower arm with the base hanging off joints
    // 2→1; pin the base link to the ground so the arm moves, not the base.
    anchorLink: 'scs215_v5',
    cameraDistance: 0.9,
    // profile joint → {urdf_joint, sign, offset(profile units)}; the offsets
    // put the vendor rest pose at the URDF's assembly zero (eyeballed).
    joints: {
      base_yaw: { urdf_joint: '1', sign: 1, offset: 1.8 },
      base_pitch: { urdf_joint: '2', sign: 1, offset: -28.9 },
      elbow_pitch: { urdf_joint: '3', sign: -1, offset: -27.6 },
      wrist_roll: { urdf_joint: '4', sign: 1, offset: -8.2 },
      wrist_pitch: { urdf_joint: '5', sign: -1, offset: 62.4 },
    },
  },
  reachy_mini: {
    urdf: '../robots/reachy_mini/vendor/urdf/robot_no_collision.urdf',
    packages: { assets: '../robots/reachy_mini/vendor/urdf/assets' },
    note: 'dev stand-in: Pollen vendor URDF (Apache-2.0) — parallel Stewart head is passive, so only body_yaw + antennas move',
    cameraDistance: 0.6,
    joints: {
      body_yaw: { urdf_joint: 'yaw_body', sign: 1, offset: 0 },
      antenna_left: { urdf_joint: 'left_antenna', sign: 1, offset: 0 },
      antenna_right: { urdf_joint: 'right_antenna', sign: -1, offset: 0 },
    },
  },
};
