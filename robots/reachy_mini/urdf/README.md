# `reachy_mini.urdf` — serial visualization chain for the Reachy Mini

**Generated** by `scripts/reachy_build_urdf.py` from Pollen Robotics' official
description (`../vendor/urdf/robot_no_collision.urdf`, Apache-2.0). Do not
hand-edit; re-run the script. Meshes live in `../meshes/` (34 STLs, 4.1 MB,
decimated to 50 %, screws omitted — see `../meshes/ATTRIBUTION.md`).

Pollen's URDF is the real mechanism: a 6-motor Stewart platform whose passive
chains close on the head link (`xl_330`). Browser URDF loaders cannot solve a
closed loop and animacy's joint tables are the SDK's *control* variables, so
this file replaces the platform with a virtual chain of exactly those variables:

```
base ──body_yaw (rev z)──► body ──(fixed)──► stewart_static   (horns + rods, posed for the neutral)
base ──head_x ─ head_y ─ head_z (prismatic, m)──► head_yaw (z) ──► head_pitch (y) ──► head_roll (x) ──► head
head ──antenna_right (rev)──► antenna_right_link      head ──antenna_left (rev)──► antenna_left_link
```

Joint values are the **reachy_mini SDK's own units and signs**: metres and
radians, exactly what `create_head_pose()` / `set_target()` take. animacy's
`ROBOT.md` converts (mm, deg, animacy signs) → URDF via `urdf_sign` — that
table is the single animacy↔SDK sign contract.

## Neutral pose (all joints = 0)

| quantity | value | source |
|---|---|---|
| head frame in base frame | `translate(0, 0, 0.177)`, identity rotation | SDK `INIT_HEAD_POSE = np.eye(4)` (`reachy_mini.py`) lifted by `head_z_offset = 0.177` m (`kinematics/placo_kinematics.py`, `assets/kinematics_data.json`) before IK |
| frame axes | x forward, y robot-left, z up | vendor FK: `camera` at (+0.0395, 0, +0.0525) in the head frame, right antenna hinge at y = −0.0522, left at +0.0522 |
| Stewart motor angles at neutral | ±35.90° alternating (`stewart_1..6` = +,−,+,−,+,−) | `AnalyticalKinematics().ik(np.eye(4))`, SDK 1.9.0 |
| vendor zero config (motors = 0) | head at z = **0.1496** m, identity | yourdfpy FK of the vendor URDF; loop closure gaps < 1e-6 m |
| rod closure at the neutral | 0.001 mm error on all six rods (rod length 0.085 m) | build-script self-check |
| body_yaw origin | (0, 0, 0.0348) m, frame rotated −90° about z, axis z | vendor `yaw_body` joint (limits ±2.79 rad) |

The vendor zero configuration (motors at 0) is *not* the user's zero: the SDK
targets the head 27.4 mm higher. This URDF uses the SDK's definition because
that is what every recorded move, `goto_target` and animacy table refers to.
The `head` frame sits at the Stewart platform's moving plate — the bottom of
the shell — which is why the shell and camera appear above it (camera 5 cm up).

## Chain order and signs

* `create_head_pose` builds `Rotation.from_euler("xyz", [roll, pitch, yaw])`
  = `Rz(yaw) · Ry(pitch) · Rx(roll)`. A chain of revolutes about *moving* axes
  composes left-to-right in chain order, so the chain is **yaw → pitch → roll**
  while the joint names stay `head_roll/head_pitch/head_yaw`. Verified:
  `tests/test_reachy_urdf.py::test_euler_composition_matches_sdk` (1e-6).
* Translation joints come first and are aligned with the base frame, matching
  the SDK's `[R | t]` where `t` is in the base frame.
* **Head poses are base-relative, not body-relative.** The SDK's IK takes the
  head pose in the base frame and `body_yaw` as an independent joint
  (`ik(create_head_pose(yaw=30°), body_yaw=30°)` returns the same Stewart angles
  as the neutral). Hence the head chain hangs off `base`; `body_yaw` turns the
  body under a fixed head.
* Pitch: the SDK/ROS right-hand rule about +y makes **+pitch = nose down**;
  reachy-duplex measured on hardware (Aug 2026) that a negative SDK pitch looks
  up, and Pollen's library agrees (`downcast1` mean pitch +16.7°, `laughing1`
  −15.5°). animacy's `head_pitch` is +up, so `ROBOT.md` sets `urdf_sign: -1`.
* Antennas: the vendor model's hinge sign is the **negative of the real robot**
  — the SDK's own MuJoCo backend writes `ctrl = -target` and reads `-qpos`
  (`daemon/backend/mujoco/backend.py`). This URDF flips both axes to
  `0 0 -1` so its values are real-robot radians. The two hinges are mirror
  images, so on the SDK `right = -left` is a symmetric gesture (`SLEEP =
  [-3.05, 3.05]`, `INIT = [-0.1745, 0.1745]`, order `[right, left]`). animacy
  defines + = *outward/down, away from the midline* for both; `ROBOT.md` sets
  `urdf_sign: -1` on `antenna_right`. 0 = vertical; the library's amazed /
  surprised ≈ +40…+120°, sad / sleep ≈ +150…+175°.

## What is static

`stewart_static` (six motor horns with arms and balls, six rods) is posed for
the neutral head and fixed to `body`. It does **not** follow the head: when the
head moves the rods visibly detach from the shell, and when `body_yaw` turns,
the horns turn with the body while the head stays. The vendor's `head_frame`,
`camera_frame`, `camera_optical` and `closing_*` helper frames are not carried.

## Verified in sim (look at `preview/*.png`)

Rendered by `scripts/reachy_render_clip.py` (matplotlib, no GL; the pose is
pushed through `ROBOT.md` exactly like the retarget pipeline):

| file | pose (animacy units) | what to see |
|---|---|---|
| `00_rest.png` | all zero | head centred over the body, antennas vertical, gaze along +x |
| `01_head_yaw_p40.png` | `head_yaw=+40` | gaze swings to +y = robot's LEFT |
| `02_head_pitch_p25.png` | `head_pitch=+25` | gaze tips UP |
| `03_head_roll_p20.png` | `head_roll=+20` | robot's right side (blue antenna) drops |
| `04_antennas_p90.png` | both antennas +90 | both ears horizontal, outward (blue right → −y, red left → +y) |
| `05_body_yaw_p60.png` | `body_yaw=+60` | body turns, head and gaze unchanged, rods detach |
| `06_head_xyz_p20mm.png` | `head_x/y/z=+20` | head shifts forward / left / up |
| `clip_amazed1_*.png` | native clip frames | one ear out, head rolled — Pollen's "amazed" |

Colours in the previews: red = left antenna, blue = right antenna, magenta =
gaze ray from the camera (dot = SDK head frame origin); world axes x red,
y green, z blue.

## Not verified on hardware

* `head_roll`, `head_x/y/z`, `body_yaw` directions follow from the vendor
  frames and the right-hand rule only.
* The antenna sign is inferred from the MuJoCo backend's negation and from the
  physical plausibility of the library's sleep pose (drooped outward, not
  through the head). Confirm on a unit: `set_target(antennas=[0, 0.8])` must
  swing the **left** antenna outward.
* `head_yaw` (+ = left) and the pitch sign are hardware-confirmed by
  reachy-duplex (Aug 2026); the 0.177 m rest height is the SDK constant, not a
  measurement.

## Regenerate / check

```
python scripts/reachy_build_urdf.py                # URDF + meshes (needs yourdfpy, trimesh, fast-simplification; SDK optional)
python scripts/reachy_render_clip.py               # preview/*.png sign checks
python scripts/reachy_render_clip.py --clip robots/reachy_mini/clips/native/amazed1.json --frames 3
python -m pytest tests/test_reachy_urdf.py -q
python -m animacy.cli check robots/reachy_mini
```
