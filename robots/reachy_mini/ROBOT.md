---
schema: animacy.robot.v1
name: reachy_mini
display_name: Reachy Mini
vendor: Pollen Robotics / Hugging Face
homepage: https://github.com/pollen-robotics/reachy_mini
license: Apache-2.0                # reachy_mini description + meshes (vendor/) are Apache-2.0
rate_hz: 30

description:
  urdf: urdf/reachy_mini.urdf      # serial visualization chain (see urdf/README.md); vendor/ has the full parallel-mechanism URDF
  mesh_scale: 1.0
  up_axis: z
  viewer:
    camera_distance: 0.6
    ground: true

# Joint names follow the reachy_mini SDK / Autonomous OS reachy driver:
# head pose = translation (mm) + roll/pitch/yaw (deg, ROS-style: +yaw left, +pitch UP), body_yaw, two antennas.
# Limits are deliberately inside the mechanism's range: they are the p99 of Pollen's
# own 85-emotion library (measured in reachy-duplex), which is the range the body
# was authored to look good in. Speeds are conservative and UNVERIFIED on hardware.
joints:
  - { name: head_x,        unit: mm,  min: -25,  max: 25,  rest: 0, max_speed: 150 }
  - { name: head_y,        unit: mm,  min: -25,  max: 25,  rest: 0, max_speed: 150 }
  - { name: head_z,        unit: mm,  min: -15,  max: 25,  rest: 0, max_speed: 150 }
  - { name: head_roll,     unit: deg, min: -30,  max: 30,  rest: 0, max_speed: 200 }
  - { name: head_pitch,    unit: deg, min: -30,  max: 30,  rest: 0, max_speed: 200, urdf_sign: -1 }   # visualization only: the URDF joint is daemon +pitch = nose DOWN; animacy + = UP
  - { name: head_yaw,      unit: deg, min: -60,  max: 60,  rest: 0, max_speed: 200 }
  - { name: body_yaw,      unit: deg, min: -120, max: 120, rest: 0, max_speed: 150 }
  - { name: antenna_left,  unit: deg, min: -140, max: 140, rest: 0, max_speed: 600 }
  - { name: antenna_right, unit: deg, min: -140, max: 140, rest: 0, max_speed: 600 }   # daemon values, plain; hinges are mirror images (see prose)

retarget:
  # A human head in conversation is a 6-DoF signal and Reachy's head is a 6-DoF
  # Stewart platform: near 1:1. Brows → antennas transfers *function* (each is
  # its owner's most legible affect channel), not geometry.
  # Head/body gains were FITTED to Pollen's emotion library by
  # scripts/retarget_fit.py: per joint, the |.|p95 excursion of the retargeted
  # human corpus is matched to 1.4x the |.|p95 of the 16 native clips (the
  # blind grader called the 1.15x fit "small"), capped by the joint bounds
  # (headroom 1.0; every head joint sits on its bound, the soft limit absorbs
  # the top) and by 2x the library's velocity p95 (docs/RETARGET.md has the
  # before/after tables). Trackers are `spring`s (docs/RETARGET.md §spring):
  # head zeta 0.7, translations and body critically damped, antennas zeta 0.45
  # (they bounce and settle, ~20% overshoot). `idle` (§idle) sways the antennas
  # and breathes the head while the person is still; `soft_limit` (§soft) is a
  # tanh knee over the last 15% of each range.
  default:
    head_yaw:   { from: head_yaw, gain: 1.7981, deadband: 0.3, spring: { hz: 4.0, zeta: 0.6 }, soft_limit: 0.15, idle: { amp: 1.0, hz: 0.15 }, settle: { seconds: 0.6 } }  # fitted by scripts/retarget_fit.py 2026-08-27
    head_pitch: { from: head_pitch, gain: 2.097, deadband: 0.3, spring: { hz: 4.0, zeta: 0.6 }, soft_limit: 0.15, idle: { amp: 1.0, hz: 0.22 }, settle: { seconds: 0.6 } }  # fitted by scripts/retarget_fit.py 2026-08-27
    head_roll:  { from: head_roll, gain: 1.7338, deadband: 0.3, spring: { hz: 4.0, zeta: 0.6 }, soft_limit: 0.15, idle: { amp: 0.8, hz: 0.17 }, settle: { seconds: 0.6 } }  # fitted by scripts/retarget_fit.py 2026-08-27
    # Whole-body participation, from Pollen's library (pooled regression on the
    # 16 clips, values around the library median): the head moves forward and
    # up when it pitches up (head_x ≈ +0.31 mm/deg, head_z ≈ +0.23 mm/deg of
    # daemon pitch) and the body follows big turns (body_yaw ≈ 0.65 × head_yaw,
    # r 0.69) — a nod is a bob, a look is a turn.
    head_x:
      mix:
        - { from: head_x, gain: 0.17955 }  # fitted by scripts/retarget_fit.py 2026-08-27
        - { from: head_pitch, gain: 0.61044 }   # 0.31 × the head's 1.617  # fitted by scripts/retarget_fit.py 2026-08-27
      spring: { hz: 3.0, zeta: 0.8 }
      idle: { amp: 1.5, hz: 0.2 }
      soft_limit: 0.15
      settle: { seconds: 0.6 }
    head_y:     { from: head_y, gain: 0.23165, spring: { hz: 3.0, zeta: 0.8 }, soft_limit: 0.15, settle: { seconds: 0.6 } }  # fitted by scripts/retarget_fit.py 2026-08-27
    head_z:
      mix:
        - { from: head_z, gain: 0.22555 }  # fitted by scripts/retarget_fit.py 2026-08-27
        - { from: head_pitch, gain: 0.3288 }   # 0.23 × the head's 1.617  # fitted by scripts/retarget_fit.py 2026-08-27
      spring: { hz: 3.0, zeta: 0.8 }
      idle: { amp: 2.0, hz: 0.22 }
      soft_limit: 0.15
      settle: { seconds: 0.6 }
    body_yaw:
      mix:
        - { from: torso_yaw, gain: 1.5858 }  # fitted by scripts/retarget_fit.py 2026-08-27
        - { from: head_yaw, gain: 0.99373 }   # 0.65 × the head's 1.687: big turns bring the body along  # fitted by scripts/retarget_fit.py 2026-08-27
      spring: { hz: 2.0, zeta: 0.9 }
      soft_limit: 0.15
      settle: { seconds: 0.6 }
    # Antennas. The two hinges are MIRROR images on the vendor URDF/daemon
    # (+right = outward, +left = inward, both toward −y), so a symmetric "ears
    # out" is antenna_left = −antenna_right — Pollen's whole library is
    # authored that way, and so is this mapping: every expressive term has
    # opposite signs on the two joints (a symmetric brow raise splays both ears
    # OUTWARD), while the head-roll term has the same sign on both (a
    # common-mode tilt). Anchors fitted from the library's statistics
    # (docs/RETARGET.md §antennas; splay = (right − left)/2, degrees):
    #   attentive1/amazed1/curious1 hold a splay of 31/49/16 (p95 45/55/49)
    #     → a full brow raise = 55 of splay;
    #   sad1/confused1/boredom1 droop to 87/81/74 (p95 130/112/131)
    #     → a full brow furrow = 85 (ears down);
    #   laughing1/yes1/cheerful1 perk 14/16/21 → a full mouth open = 15;
    #   head down drags the ears down: pooled slope −2.2 splay/deg of pitch,
    #     damped to −0.8 so ordinary nods only flutter them;
    #   common-mode tilt vs head roll: pooled slope −0.67, within-clip −0.47 → −0.5 (counter-rotation).
    # Hardware array order ([left, right] on this unit vs the SDK's
    # [right, left]) is the sink's business, not this mapping's: the signs
    # here are per named joint.
    antenna_left:
      mix:
        - { from: brow_l, gain: -55 }
        - { from: brow_furrow, gain: -85 }
        - { from: mouth_open, gain: -15 }
        - { from: head_pitch, gain: 0.8 }
        - { from: head_roll, gain: -0.5 }
      spring: { hz: 3.5, zeta: 0.45 }
      idle: { amp: 6.0, hz: 0.3 }
      soft_limit: 0.15
      settle: { seconds: 0.6 }
    antenna_right:
      mix:
        - { from: brow_r, gain: 55 }
        - { from: brow_furrow, gain: 85 }
        - { from: mouth_open, gain: 15 }
        - { from: head_pitch, gain: -0.8 }
        - { from: head_roll, gain: -0.5 }
      spring: { hz: 3.5, zeta: 0.45 }
      idle: { amp: 6.0, hz: 0.3 }
      soft_limit: 0.15
      settle: { seconds: 0.6 }
  # Puppet mode: the hand is the head (fist-bump / high-five behaviours).
  puppet:
    head_yaw:   { from: shoulder_yaw, gain: 0.6, smooth_hz: 6 }
    head_pitch: { from: wrist_pitch, gain: 0.5, smooth_hz: 6 }
    head_roll:  { from: wrist_roll,  gain: 0.4, smooth_hz: 6 }
    head_z:     { from: shoulder_pitch, gain: 0.25, offset: -15, smooth_hz: 4 }
    head_x:     { from: elbow_flex, gain: -0.2, offset: 20, smooth_hz: 4 }
    antenna_left:  { from: hand_open, gain: 100, smooth_hz: 8 }
    antenna_right: { from: hand_open, gain: 100, smooth_hz: 8 }

export:
  formats: [pollen_move, lerobot]

runtime:
  kind: reachy_sdk
  url: http://192.168.1.60:8000    # the daemon on the Wireless unit; Lite runs it on the laptop
  stream_hz: 30
  extra:
    sdk_call: set_target            # ReachyMini.set_target(head=4x4, antennas=[r, l] rad, body_yaw=rad)

native_clips:
  dir: clips/native
  format: json                     # converted from Pollen's emotion library (scripts/pollen_npz_to_joints.py)

---

# Reachy Mini

Pollen Robotics' desk robot (Hugging Face): a 6-DoF Stewart-platform head on a
rotating body with two antenna "ears", a camera, a 4-mic array and a speaker.
It is also an official body in Autonomous OS (`robots/reachy-mini`,
`hal/drivers/motors/reachy_service.py`), which is where these joint names come
from — a clip retargeted here plays on either stack.

## Sign conventions

**Hardware-verified 2026-08-26** on the physical Reachy Mini Wireless
(192.168.1.60, owner watching): `docs/evidence/reachy_sim2real_20260826.md`,
produced by `scripts/reachy_sim2real.py` streaming this profile's `default`
mapping through `animacy.sinks.ReachyDaemonSink` (`POST /api/move/set_target`).

**What the daemon receives is the plain profile value** — deg→rad, mm→m,
`target_antennas = [antenna_left, antenna_right]` — with exactly one sign
change, in the sink: `pitch` is negated (daemon `+pitch` = nose down). Nothing
else flips anywhere. The `urdf_sign` column in the front matter is for the
**visualization URDF only** (`urdf/reachy_mini.urdf` models the daemon's
convention: its `head_pitch` joint is +down, its antennas take daemon values
as-is), so `head_pitch: urdf_sign -1` is the viewer's copy of the sink's flip,
not a second flip on the way to the robot. The sink never reads `urdf_sign`.

| joint | animacy `+` means | sent to the daemon | evidence |
|---|---|---|---|
| `head_yaw` | robot turns to ITS left | `+yaw` rad | hardware: "look left" turned to the robot's left |
| `head_pitch` | look **UP** | `-pitch` rad (the sink's one flip) | hardware: "look up" looked up; library `downcast1` mean +16.7°, `laughing1` −15.5° (daemon frame) |
| `head_roll` | robot's right ear drops | `+roll` rad | hardware: "roll, right ear down" dropped the right ear |
| `head_x/y/z` | forward / left / up, mm | metres, base frame | tracked by the daemon (`lean in`); direction not eyeballed |
| `body_yaw` | body turns to its left | `+rad` | hardware: "turn body left" turned left; head yaw read back ≈ 0 while the body sat at +30° |
| `antenna_left` | `target_antennas[0]`, plain | rad | hardware: "left brow only" (+90 on `[0]`) moved the robot's LEFT antenna |
| `antenna_right` | `target_antennas[1]`, plain | rad | tracked by the daemon (+90 → 61° at the mid-hold sample) |

- **Antennas are mirror hinges.** In the vendor URDF (byte-identical to what
  the daemon serves at `/api/kinematics/urdf`) `+right_antenna` swings the
  right antenna outward (toward −y) and `+left_antenna` swings the left antenna
  inward — also toward −y. Equal values on both are therefore *not* a symmetric
  gesture; a symmetric "ears out" is `antenna_left = -antenna_right`. Pollen's
  whole library is authored that way (`SLEEP = [-3.05, +3.05]`; `yes1` left
  −18…−2 / right +13…+29; `amazed1` left −19 / right +39…+91; converted clips
  in `clips/native/` keep the daemon's values). The `default` mapping above
  sends the same sign to both (`brow_* × 90`), which on hardware sweeps both
  antennas toward the robot's right — negate the `antenna_left` gains if
  brows-up should read as "ears open". The out/in geometry itself has not been
  eyeballed on hardware: send `target_antennas = [-1.0, +1.0]` — both antennas
  should splay outward; if they cross instead, flip both signs in the mapping.
- **Which motor is `[0]`** is a per-unit fact: the SDK's `hardware_config.yaml`
  lists motor id 17 as `right_antenna` before `left_antenna`, and its MuJoCo
  backend comments say `[right, left]` — yet this unit visibly moved the LEFT
  antenna for element `[0]`. If another unit differs, swap the two joints in the
  mapping, never in captured data.
- **Head poses are base-relative, not body-relative.** The SDK solves the head
  pose in the base frame with `body_yaw` as an independent joint
  (`ik(create_head_pose(yaw=30°), body_yaw=30°)` returns the neutral Stewart
  angles; the daemon read back head yaw ≈ 0 with the body at +30°), and the
  URDF hangs the head chain off `base` accordingly. So the `0.25·head_yaw`
  term in `retarget.default.body_yaw` does not over-rotate the head; it only
  brings the body along underneath it. The daemon's `present_head_pose` is this
  offset from the neutral head frame (≈ 0 after `wake_up`).
- If your unit differs on any row, fix it with `gain: -1` in the mapping (never
  in captured data) and say so in an issue.

## Neutral pose

Head level and centred 0.177 m above the base frame, antennas vertical (0),
body facing forward: the SDK's `INIT_HEAD_POSE` (identity lifted by
`head_z_offset`), reached with `goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS)`
or `POST /api/move/goto` with an all-zero pose. After `wake_up` the daemon's
`present_head_pose` reads ≈ 0 on every axis (measured 2026-08-26).

## What each canonical channel means on this body

- head 6-DoF → head 6-DoF, scaled to the emotion library's envelope (the model
  learns the range it is shown; a policy trained against hardware limits slams them).
- brows → antennas, mouth-open adds a flick while speaking.
- torso yaw → body yaw; big head turns bring the body along.
- `puppet`: hand pose → head pose, hand open/close → antennas.

## Getting frames onto the real robot

```
animacy retarget --robot reachy_mini --clip data/clips/<clip> -o out/hello.json --format pollen_move
python scripts/play_pollen_move.py --host 192.168.1.60 out/hello.json
```
`animacy serve --robot reachy_mini` streams `set_target` at 30 Hz.

## Verification checklist

```
animacy check robots/reachy_mini
animacy profile export robots/reachy_mini -o web/robots/reachy_mini.json
```
Web viewer → Reachy Mini → play `native/amazed1` (antennas up, head back), then
a captured clip: "look left" turns left, "look up" tips up, brows up → antennas up.
