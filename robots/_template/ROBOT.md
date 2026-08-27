---
schema: animacy.robot.v1
name: myrobot                    # slug: [a-z][a-z0-9_]*
display_name: My Robot
vendor: Someone
homepage: https://example.com
license: Apache-2.0              # of the URDF + meshes you ship in this folder
rate_hz: 30

description:
  urdf: urdf/myrobot.urdf        # relative to this file
  mesh_scale: 1.0
  up_axis: z
  viewer:
    camera_distance: 0.8
    ground: true

joints:
  - { name: neck_yaw,   unit: deg, min: -90, max: 90, rest: 0, max_speed: 120 }
  - { name: neck_pitch, unit: deg, min: -45, max: 45, rest: 0, max_speed: 120 }
  # urdf_joint / urdf_sign / urdf_offset are optional (default: same name, +1, 0)

retarget:
  default:
    neck_yaw:   { from: head_yaw }
    neck_pitch: { from: head_pitch, gain: 1.0, deadband: 0.5, smooth_hz: 6 }

export:
  formats: [csv]

runtime:
  kind: none

---

# My Robot

## Sign conventions (UNVERIFIED — mark each line once checked on hardware)

- `neck_yaw` + = ?  (canonical `head_yaw` + = subject's left; if the robot turns the other way set `gain: -1`)
- `neck_pitch` + = ?

## Neutral pose

What the robot looks like at `rest`.

## What each channel means on this body

- gaze (`head_yaw`, `head_pitch`) → the joints that point the "face".
- lean (`torso_lean_fwd`, `head_x`) → base joints, if any.
- affect (`brow_*`) → the body's most legible expressive channel (ears, antennas, a head tip-up).

## Getting frames onto the real robot

Endpoint / SDK, format, rate, and the safety ceiling `max_speed` comes from.

## Verification checklist

```
animacy check robots/myrobot
animacy profile export robots/myrobot
animacy retarget --robot myrobot --clip data/clips/<any> -o /tmp/out.csv
```
Open the web viewer, load the clip, confirm "look up" looks up and "turn left" turns left.
