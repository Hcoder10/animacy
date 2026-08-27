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
  - { name: head_pitch,    unit: deg, min: -30,  max: 30,  rest: 0, max_speed: 200, urdf_sign: -1 }   # SDK/ROS +pitch = nose DOWN; animacy + = UP
  - { name: head_yaw,      unit: deg, min: -60,  max: 60,  rest: 0, max_speed: 200 }
  - { name: body_yaw,      unit: deg, min: -120, max: 120, rest: 0, max_speed: 150 }
  - { name: antenna_left,  unit: deg, min: -140, max: 140, rest: 0, max_speed: 600 }
  - { name: antenna_right, unit: deg, min: -140, max: 140, rest: 0, max_speed: 600, urdf_sign: -1 }   # mirror axis: SDK right + = inward; animacy + = outward for both

retarget:
  # A human head in conversation is a 6-DoF signal and Reachy's head is a 6-DoF
  # Stewart platform: near 1:1. Brows → antennas transfers *function* (each is
  # its owner's most legible affect channel), not geometry.
  default:
    head_yaw:   { from: head_yaw,   gain: 0.8, deadband: 0.3, smooth_hz: 6 }
    head_pitch: { from: head_pitch, gain: 0.8, deadband: 0.3, smooth_hz: 6 }
    head_roll:  { from: head_roll,  gain: 0.8, deadband: 0.3, smooth_hz: 6 }
    head_x:     { from: head_x, gain: 0.2, smooth_hz: 4 }
    head_y:     { from: head_y, gain: 0.2, smooth_hz: 4 }
    head_z:     { from: head_z, gain: 0.25, smooth_hz: 4 }
    body_yaw:
      mix:
        - { from: torso_yaw, gain: 0.8 }
        - { from: head_yaw,  gain: 0.25 }   # big turns bring the body along
      smooth_hz: 3
    antenna_left:
      mix:
        - { from: brow_l,    gain: 90 }
        - { from: head_roll, gain: -0.5 }
        - { from: mouth_open, gain: 15 }
      smooth_hz: 8
    antenna_right:
      mix:
        - { from: brow_r,    gain: 90 }
        - { from: head_roll, gain: 0.5 }
        - { from: mouth_open, gain: 15 }
      smooth_hz: 8
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

`urdf/reachy_mini.urdf` uses the SDK's own units and signs (`create_head_pose`:
metres, radians, `Rz(yaw)·Ry(pitch)·Rx(roll)`; antennas as
`set_target(antennas=[right, left])` radians), so the `urdf_sign` column in the
front matter **is the animacy↔SDK sign table**. Everything below is verified in
simulation (`scripts/reachy_render_clip.py` → `urdf/preview/*.png`,
`tests/test_reachy_urdf.py`) and cross-checked against Pollen's emotion
library; hardware evidence is listed where it exists (details and sources in
`urdf/README.md`).

| joint | animacy `+` means | SDK value | evidence |
|---|---|---|---|
| `head_yaw` | turn toward the robot's LEFT (+y) | `+yaw` | hardware (reachy-duplex, Wireless unit, Aug 2026) |
| `head_pitch` | look **UP** | `-pitch` (`urdf_sign: -1`) — ROS/SDK `+pitch` is nose-down | hardware (Aug 2026: negative SDK pitch looks up); library `downcast1` mean +16.7°, `laughing1` −15.5° |
| `head_roll` | right ear drops toward the right shoulder | `+roll` | right-hand rule about +x; **unverified on hardware** |
| `head_x/y/z` | forward / left / up, mm | metres, same base frame | vendor frames (camera at +x, right antenna at −y); **unverified on hardware** |
| `body_yaw` | body turns left | `+body_yaw` | vendor `yaw_body` axis; **unverified on hardware** |
| `antenna_left` | swings outward/down, away from the midline; 0 = vertical | `+left` | see below; **unverified on hardware** |
| `antenna_right` | same, mirror | `-right` (`urdf_sign: -1`) | see below; **unverified on hardware** |

- **Head poses are base-relative, not body-relative.** The SDK solves the head
  pose in the base frame with `body_yaw` as an independent joint
  (`ik(create_head_pose(yaw=30°), body_yaw=30°)` returns the neutral Stewart
  angles), and the URDF hangs the head chain off `base` accordingly. So the
  `0.25·head_yaw` term in `retarget.default.body_yaw` does not over-rotate the
  head; it only brings the body along underneath it.
- **Antennas.** The two hinges are mirror images, so on the SDK a symmetric
  gesture is `right = -left` (the whole library does this; `SLEEP = [-3.05,
  3.05]`, `INIT = [-0.1745, 0.1745]`). animacy flips the right one so equal
  values are symmetric and `+` is "outward/down" on both. The antennas rest
  vertical, so the expressive range is 0…+180: amazed/surprised ≈ +40…+120°,
  sad/sleep ≈ +150…+175°; "raise" has nowhere to go, "open" is the gesture.
  The vendor model's hinge sign is the negative of the real robot's (the SDK's
  MuJoCo backend writes `ctrl = -target`); the URDF axes are already flipped to
  real-robot sign. To confirm on a unit: `set_target(antennas=[0, 0.8])` must
  swing the **left** antenna outward.
- **Exporting to the SDK** must apply the same table: head =
  `create_head_pose(head_x/1000, head_y/1000, head_z/1000, head_roll,
  -head_pitch, head_yaw, degrees=True)`, `antennas = [-radians(antenna_right),
  radians(antenna_left)]` (order `[right, left]`), `body_yaw` in radians.
  `scripts/pollen_npz_to_joints.py` is the inverse of exactly this.
- If your unit differs on any row, fix it with `gain: -1` in the mapping (never
  in captured data) and say so in an issue.

## Neutral pose

Head level and centered over the body, antennas relaxed at 0, body facing
forward — `goto_zero()` in the SDK.

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
