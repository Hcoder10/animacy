---
schema: animacy.robot.v1
name: so101
display_name: SO-101 arm
vendor: TheRobotStudio / LeRobot (Hugging Face)
homepage: https://github.com/TheRobotStudio/SO-ARM100
license: Apache-2.0                # SO-ARM100 repo (URDF + STL meshes), see meshes/ATTRIBUTION.md
rate_hz: 30

description:
  urdf: urdf/so101.urdf            # vendor's so101_new_calib.urdf, only mesh paths rewritten
  mesh_scale: 1.0
  up_axis: z
  viewer:
    camera_distance: 0.9
    ground: true

# Joint names are LeRobot's SO-101 motor names (`so101_follower`, bus order 1..6).
# Units: degrees (LeRobot `use_degrees=True`). Limits are the URDF's own
# (so101_new_calib.urdf, rad -> deg, rounded inward). At all-zero the arm is
# straight out horizontally along +x; `rest` is an "attentive" pose: upper arm
# vertical, forearm forward-down, gripper level and looking ~15 deg up.
# max_speed is conservative and UNVERIFIED: the STS3215 can do ~250 deg/s
# (Autonomous OS measured it on the same servo), LeRobot ships no safety file.
joints:
  - { name: shoulder_pan,  unit: deg, min: -109, max: 109, rest: 0,   max_speed: 120 }   # URDF +-1.91986 rad = +-110.0
  - { name: shoulder_lift, unit: deg, min: -99,  max: 99,  rest: -90, max_speed: 120 }   # URDF +-1.74533 rad = +-100.0
  - { name: elbow_flex,    unit: deg, min: -96,  max: 96,  rest: 45,  max_speed: 120 }   # URDF +-1.69 rad = +-96.8
  - { name: wrist_flex,    unit: deg, min: -94,  max: 94,  rest: 30,  max_speed: 150 }   # URDF +-1.65806 rad = +-95.0
  - { name: wrist_roll,    unit: deg, min: -157, max: 162, rest: 0,   max_speed: 150 }   # URDF -2.74385..2.84121 rad = -157.2..162.8
  - { name: gripper,       unit: deg, min: -9,   max: 99,  rest: 10,  max_speed: 200 }   # URDF -0.174533..1.74533 rad = -10..100 (LeRobot 0 = closed, 100 = open)

retarget:
  # Talking-head mode: the gripper is the "face". URDF signs (yourdfpy FK, see
  # prose): +shoulder_pan swings the tip to the robot's RIGHT; +shoulder_lift,
  # +elbow_flex and +wrist_flex all move the tip DOWN. Hence the negative gains
  # on gaze. Gains are seeded so a big human move stays inside the joint range
  # from `rest` (yaw 90 deg -> 72 deg of pan; pitch 60 -> 36 of wrist).
  default:
    shoulder_pan: { from: head_yaw, gain: -0.8, deadband: 0.3, smooth_hz: 6 }
    wrist_flex:
      mix:
        - { from: head_pitch, gain: -0.6 }   # look up = tip the gripper up
        - { from: brow_l,     gain: -6 }     # brows up = a small tip-up
        - { from: brow_r,     gain: -6 }
      deadband: 0.3
      smooth_hz: 6
    shoulder_lift:
      mix:
        - { from: torso_lean_fwd, gain: 0.6 }   # lean in = shoulder comes forward/down
        - { from: head_x,         gain: 0.1 }
        - { from: head_z,         gain: -0.1 }  # taller = shoulder lifts
      smooth_hz: 4
    elbow_flex:
      mix:
        - { from: torso_lean_fwd, gain: -0.4 }  # lean in = arm extends (less elbow)
        - { from: head_x,         gain: -0.1 }
      smooth_hz: 4
    wrist_roll: { from: head_roll, gain: 0.6, smooth_hz: 6 }
    gripper:
      mix:
        - { from: mouth_open, gain: 30 }        # gripper flutters with speech
        - { from: smile,      gain: 10 }
      smooth_hz: 8
  # Puppet mode: the canonical (right) arm chain 1:1. joint = rest + offset + gain*channel,
  # offsets cancel `rest` so the human's joint angles land on the robot's directly.
  puppet:
    shoulder_pan:  { from: shoulder_yaw,   gain: -1 }                      # +yaw = arm swings left; +pan = right
    shoulder_lift: { from: shoulder_pitch, gain: -1, offset: 180, max: 40 } # pitch 0 (hanging) -> +90 (clamped 40, table); 90 -> 0; 180 -> -90 (up)
    elbow_flex:    { from: elbow_flex,     gain: -1, offset: -45 }         # 0 = straight on both; URDF + folds the forearm DOWN (storage side), a human elbow folds the hand UP
    wrist_flex:    { from: wrist_pitch,    gain: -1, offset: -30 }         # + hand tips up = -wrist_flex
    wrist_roll:    { from: wrist_roll,     gain: 1 }                       # sign UNVERIFIED
    gripper:       { from: hand_open,      gain: 100, offset: -10 }        # fist 0 -> closed, spread 1 -> 100

export:
  formats: [lerobot]

runtime:
  kind: none

---

# SO-101 arm

TheRobotStudio's 5-DoF + gripper desk arm, the standard LeRobot follower
(`so101_follower`). Autonomous OS declares the same body in `robots/so101`.
The URDF is Pollen-style onshape-to-robot output from the SO-ARM100 repo,
used unmodified except for mesh paths; meshes are decimated copies
(`meshes/ATTRIBUTION.md`).

## Sign conventions (verified in sim only — nothing here has touched hardware)

From yourdfpy FK on the vendor URDF (`dev/render_previews.py`,
`tests/test_so101.py`): the base frame is x forward, y left, z up; at all-zero
the arm lies straight out along +x with the gripper 0.39 m out, 0.23 m up.

- `shoulder_pan` + = the arm swings to the robot's **right** (tip toward −y),
  so canonical `head_yaw`/`shoulder_yaw` (+ = left) use `gain: -1`.
- `shoulder_lift` + = the upper arm rotates **down** from horizontal (−90 =
  straight up). LeRobot's folded "rest" is around −100/+100 on lift/elbow.
- `elbow_flex` + = the forearm bends **down** relative to the upper arm
  (0 = straight) — the side it folds to for storage. A human elbow folds the
  hand the other way, so `puppet` uses `gain: -1` (assumes the elbow can bend
  both ways, as the URDF's ±96° says; if your unit's elbow only folds one way,
  use `gain: 1`). `wrist_flex` + = the gripper tips **down**.
- `wrist_roll` + = rotation about the forearm axis; which way is unverified.
- `gripper` 0 ≈ closed, +100 = fully open (URDF range −10…100).
- All directions come from the URDF only. LeRobot's calibration can flip a
  motor's sign per unit; if "look left" turns right on yours, set `gain: -1`
  in the mapping — never in captured data.

## Neutral pose

`rest` = pan 0, lift −90, elbow 45, wrist_flex 30, roll 0, gripper 10: upper
arm vertical, forearm reaching forward-down, gripper level at ~0.28 m height
and 0.20 m forward, "looking" 15° up at a seated person. Not LeRobot's folded
rest — that one is for storage, not conversation.

## What each canonical channel means on this body

- gaze (`head_yaw`, `head_pitch`) → `shoulder_pan`, `wrist_flex`: the gripper is the face.
- lean (`torso_lean_fwd`, `head_x`, `head_z`) → `shoulder_lift` + `elbow_flex`: reach in / sit back / rise.
- affect: `brow_l/r` → a small `wrist_flex` tip-up; `mouth_open`/`smile` → `gripper` flutter.
- `head_roll` → `wrist_roll` (head tilt).
- `puppet`: the person's right arm drives the arm joint-for-joint; a fist closes the gripper.

## Getting frames onto the real robot

No live runtime here (`runtime.kind: none`). Export a LeRobot v3 dataset
(`export.formats: [lerobot]`, `animacy.lerobot_export`) and replay it with
LeRobot's `so101_follower` configured with `use_degrees=True`; without it the
bus expects `RANGE_M100_100` units and these degrees must be rescaled per the
unit's calibration. Keep the follower's own `max_relative_target` safety on.

## Verification checklist

```
animacy check robots/so101
animacy profile export robots/so101 -o web/robots/so101.json
python web/dev/build_manifest.py
python robots/so101/dev/render_previews.py        # urdf/preview/*.png: rest, look-left, puppet wave
python -m pytest tests/test_so101.py -q
```
In the viewer: "look left" swings the gripper to the robot's left, "look up"
tips it up, a lean-in extends the arm, speech flutters the gripper.
