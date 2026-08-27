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
  - { name: head_pitch,    unit: deg, min: -30,  max: 30,  rest: 0, max_speed: 200 }
  - { name: head_yaw,      unit: deg, min: -60,  max: 60,  rest: 0, max_speed: 200 }
  - { name: body_yaw,      unit: deg, min: -120, max: 120, rest: 0, max_speed: 150 }
  - { name: antenna_left,  unit: deg, min: -140, max: 140, rest: 0, max_speed: 600 }
  - { name: antenna_right, unit: deg, min: -140, max: 140, rest: 0, max_speed: 600 }

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

- Head rotations are ROS-style in the SDK: **+yaw = left, +pitch = up? NO** —
  measured on a Wireless unit in reachy-duplex (Aug 2026): with
  `create_head_pose(..., degrees=True)`, **negative pitch looks UP**, and
  positive yaw turns toward the robot's left. Canonical `head_pitch` is +up, so
  the exporter/URDF carries the sign: `urdf_sign` stays +1 here and the Pollen
  exporter flips pitch when it builds the 4×4 (documented in `animacy/export.py`).
  If your unit differs, set `gain: -1` on `head_pitch` and say so in an issue.
- Antennas: SDK order is `[right, left]` in radians; positive raises the antenna
  forward/up on the vendor's emotion library. `antenna_left`/`antenna_right`
  are in degrees here, converted by the exporter.
- `body_yaw` positive = body turns left (same sense as `head_yaw`).

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
