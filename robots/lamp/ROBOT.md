---
schema: animacy.robot.v1
name: lamp
display_name: Autonomous Lamp
vendor: Autonomous
homepage: https://github.com/autonomous-ai/autonomous-os/tree/main/robots/lamp
license: Apache-2.0                # Autonomous OS robots/ tree (CAD + recordings) is Apache-2.0
rate_hz: 30                        # HAL playback loop (HAL_SERVO_FPS)

description:
  urdf: urdf/lamp.urdf
  mesh_scale: 1.0
  up_axis: z
  viewer:
    camera_distance: 0.9
    ground: true

# Servo order and names are Autonomous OS's own (hal/models.py ServoMoveRequest):
# IDs 1..5 = base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch, degrees,
# API-clamped to ±90. `rest` is the median of the vendor's idle.csv; the
# `native_clips` envelope (p1..p99 over all 31 vendor clips) is:
#   base_yaw -56..42 | base_pitch 0..74 | elbow_pitch -1..62 | wrist_roll -50..67 | wrist_pitch -82..28
# max_speed: 250 deg/s is the measured STS3215 ceiling used by the vendor's
# recording playback (hal/drivers/motors/recording_timing.py). Their SAFETY.md
# motion.max_speed=120 applies to commanded /servo/move paths, which the live
# runtime respects separately (runtime.stream_max_speed).
# UNITS CAVEAT: the vendor drives the bus in LeRobot RANGE_M100_100 mode
# (hal/follower/config_hal_follower.py use_degrees=False), so a clip value is a
# fraction of the per-unit calibrated span (~1.07 deg/unit yaw, ~1.16 roll; pitch
# joints calibration-dependent). Values here are those vendor units, labelled
# `deg` because they are within ~10% of degrees and the viewer needs an angle;
# directions and topology are exact, amplitudes approximate until a unit is measured.
joints:
  - { name: base_yaw,    unit: deg, min: -90, max: 90, rest: -1.8, max_speed: 250 }
  - { name: base_pitch,  unit: deg, min: -90, max: 90, rest: 28.9, max_speed: 250 }
  - { name: elbow_pitch, unit: deg, min: -90, max: 90, rest: 27.6, max_speed: 250 }
  - { name: wrist_roll,  unit: deg, min: -90, max: 90, rest:  8.2, max_speed: 250 }
  - { name: wrist_pitch, unit: deg, min: -90, max: 90, rest: -62.4, max_speed: 250 }

retarget:
  # Talking-head mode: the person's face and torso drive the lamp like a
  # curious creature. Gains were seeded from the vendor's own clips (a
  # "headshake" is ~117° of wrist_roll + 39° of base_yaw; a "nod" is ~12° base
  # + 26° elbow + 17° wrist_pitch; "sad" droops 32/45/64) so the retarget lands
  # inside the envelope the body was authored for.
  # Vendor signs (see "Sign conventions" below): +base_yaw and +wrist_roll pan
  # RIGHT, +wrist_pitch tips the head DOWN — all opposite to the canonical
  # channels (+yaw = left, +pitch = up), hence the negative gains. Rule 4 of
  # the spec: signs are fixed here, never in the URDF or the data.
  default:
    base_yaw:
      mix:
        - { from: head_yaw,  gain: -0.45 }
        - { from: torso_yaw, gain: -0.6 }
      deadband: 0.3
      smooth_hz: 5
    wrist_roll:
      # With the head pitched down toward the desk, rolling about the forearm
      # axis pans the lamp head left/right — that is the vendor's own
      # "headshake". So gaze yaw lives here, not on base_yaw.
      mix:
        - { from: head_yaw,  gain: -1.0 }
        - { from: head_roll, gain: -0.3 }
      deadband: 0.3
      smooth_hz: 6
      min: -60
      max: 70
    wrist_pitch:
      mix:
        - { from: head_pitch, gain: -0.9 }
        - { from: brow_l,     gain: -5.0 }  # brow raise = "perk up" (head tips up a touch)
        - { from: brow_r,     gain: -5.0 }
        - { from: torso_lean_fwd, gain: 0.5 }  # keep looking at the person while leaning in
      smooth_hz: 6
      min: -85
      max: 30
    base_pitch:
      mix:
        - { from: torso_lean_fwd, gain: 1.0 }
        - { from: head_x,         gain: 0.12 }
      smooth_hz: 4
      min: 0
      max: 75
    elbow_pitch:
      mix:
        - { from: head_z,         gain: 0.35 }  # rise up / droop down
        - { from: torso_lean_fwd, gain: 0.4 }
        - { from: mouth_open,     gain: 6.0 }   # a little lift while talking
      smooth_hz: 4
      min: -5
      max: 62
  # Puppet mode: your own arm IS the lamp. Shoulder → base, elbow → elbow,
  # wrist → neck/head. Offsets put a relaxed forearm-forward pose at the
  # vendor's rest.
  puppet:
    base_yaw:    { from: shoulder_yaw, gain: -1.0, smooth_hz: 6 }
    base_pitch:  { from: shoulder_pitch, gain: 0.6, offset: -25.0, smooth_hz: 6, min: 0, max: 80 }
    elbow_pitch: { from: elbow_flex, gain: 0.6, offset: -25.0, smooth_hz: 6, min: -10, max: 65 }
    wrist_roll:  { from: wrist_roll, gain: -1.0, smooth_hz: 8, min: -70, max: 85 }
    wrist_pitch: { from: wrist_pitch, gain: -1.0, smooth_hz: 8, min: -90, max: 30 }

export:
  formats: [autonomous_os_csv, lerobot]
  autonomous_os_csv:
    column_suffix: ".pos"
    timestamp_column: timestamp
    fps: 30

runtime:
  kind: autonomous_os_hal
  url: http://127.0.0.1:5001       # HAL on the lamp; LAN use needs the vendor's Phase-0 token auth
  stream_hz: 30
  extra:
    upload: /servo/upload           # multipart CSV → playable by name
    play: /servo/play               # {"recording": name}
    stream: /servo/move             # {"positions": {...}, "duration": s}, safety-gated
    stream_max_speed: 120           # SAFETY.md motion.max_speed for commanded moves
    stop: /servo/stop

native_clips:
  dir: clips/native
  format: autonomous_os_csv

---

# Autonomous Lamp

A 5-servo desk lamp (Feetech STS3215 ×5, OrangePi 4 Pro, LeLamp-derived body)
running [Autonomous OS](https://github.com/autonomous-ai/autonomous-os). Its
agent moves it by emitting `[HW:/servo/play:{"recording":"nod"}]` markers that
HAL plays from `hal/recordings/*.csv` — the 31 vendor clips in `clips/native/`
are those files verbatim (Apache-2.0, © Autonomous).

## Sign conventions

Joint values here are **identical to the vendor CSV values** (that is what
`/servo/upload` receives), so the URDF in `urdf/` is built in the vendor's
convention and `urdf_sign`/`urdf_offset` stay at identity. The URDF's
geometry, chain and joint directions come from the vendor's own CAD
(`cad_src/lamp.glb`) and the vendor's device notes; `urdf/README.md` has the
evidence for every number.

Direction of a **positive** value on each joint, taken from Autonomous's
device-measured notes (`hal/drivers/tracking/constants.py`, unit lamp-ac82,
2026-08-25) and reproduced by the URDF (`tests/test_lamp_urdf.py`):

- `base_yaw` + → the head pans to the lamp's **right** (clockwise from above).
- `wrist_roll` + → pans **right** as well; it rolls the head about the neck
  axis, and because the head looks perpendicular to the neck that swings the
  gaze sideways (the vendor's `headshake`). Note: these two are the *opposite*
  of the canonical `head_yaw` (+ = left), so a `head_yaw → base_yaw/wrist_roll`
  mapping needs a negative `gain`.
- `base_pitch` + → lower arm leans **forward**, camera tips down.
- `elbow_pitch` + → fold closes, head rises, camera tips **up** (vendor:
  "elbow +1.6 framed the desk, +54.8 the ceiling"; opposite sense to
  `base_pitch`, their `ELBOW_PITCH_SIGN = -1`).
- `wrist_pitch` + → head tips **down**; **negative = look up** (vendor: "looking
  up drives wrist NEGATIVE"). Also opposite to canonical `head_pitch` (+ = up).

Status: **verified in simulation only.** What was checked: (a) FK of the URDF
at the vendor home values reproduces the CAD assembly pose; (b) rendering the
vendor clips through `animacy.retarget.to_urdf_values`
(`scripts/lamp_render_clip.py`, PNGs in `urdf/preview/`) gives an upright lamp
looking forward/slightly down at rest, `sad`/`sleepy` drooping to the desk,
`nod` bobbing, `headshake` panning left-right, `stretching` the tallest pose;
(c) every clip keeps the head above the desk and in front of the base. Two
things are **not** verified and are only inferable from vendor artefacts: the
vendor value of the CAD pose (taken as the clips' home pose 29.8/27.1/−26.3/8.2,
see README) and the unit scale — the vendor drives the bus in LeRobot
`RANGE_M100_100` mode (`use_degrees = False`), so a value is a fraction of the
per-unit calibrated span (~1.07°/unit for yaw, ~1.16°/unit for roll, pitch joints
calibration-dependent); the URDF and this file treat 1 unit = 1°. Hardware
verification is pending a physical unit — if your lamp turns the wrong way on
`look left`, flip `gain` on `wrist_roll`/`base_yaw` in `default` and open an
issue with the unit id.

Physical reading of the vendor data (clip envelope, in the URDF's reading):

- `base_yaw` — turns the whole arm on the base bearing. Median −2, ±50 in use.
- `base_pitch` — lower arm; 0 = lying back on the base, ~30 at rest (arm ~45°
  back of vertical, the CAD pose), 74 = near vertical (`stretching`).
- `elbow_pitch` — upper arm fold; ~27 at rest (90° fold), 62 = folded up tall
  (`stretching`), −13 = opened out (`wake_up` start, asleep lying back).
- `wrist_roll` — roll about the neck axis; pans the head (headshake ±60).
- `wrist_pitch` — head tip; −26 = the CAD pose (looking 45° down the neck),
  −62 = idle (looking forward, slightly down), +28 = looking straight down at
  the desk, −82 = looking up.

## Neutral pose

`rest` is the median of `idle.csv`: arm slightly forward, head tipped down
toward the desk/user — the "attentive lamp" pose the whole clip library
returns to.

## What each canonical channel means on this body

- gaze: `head_yaw` → `wrist_roll` (+ a little `base_yaw`), `head_pitch` → `wrist_pitch`.
- lean: `torso_lean_fwd` + `head_x` → `base_pitch`, with a counter-term on
  `wrist_pitch` so the head keeps looking at the person while the body leans in.
- height / droop: `head_z` → `elbow_pitch`.
- affect: brow raise → a 5° up-tick of the head (`wrist_pitch`) — the lamp's most
  legible "perk up".
- talking: `mouth_open` adds a little `elbow_pitch` lift so speech has body.
- `puppet` mode: shoulder/elbow/wrist of your own arm → base/elbow/wrist 1:1.

## Getting frames onto the real lamp

```
animacy retarget --robot lamp --clip data/clips/<clip> -o out/hello.csv --format autonomous_os_csv
curl -sX POST http://<lamp>:5001/servo/upload -F "file=@out/hello.csv" -F "recording_name=hello"
curl -sX POST http://<lamp>:5001/servo/play -H 'Content-Type: application/json' -d '{"recording":"hello"}'
```

The exporter enforces the upload route's own checks (`timestamp` column,
`<joint>.pos` names, known joints, numeric values) and the 250 deg/s ceiling by
stretching time, never clipping. Live streaming uses `/servo/move` and is held
to the SAFETY.md 120 deg/s ceiling by the runtime.

## Verification checklist

```
animacy check robots/lamp
animacy profile export robots/lamp -o web/robots/lamp.json
animacy retarget --robot lamp --clip data/clips/<clip> -o /tmp/x.csv --format autonomous_os_csv
```
Open the web viewer → Lamp → play `native/nod` (should bob), `native/headshake`
(should pan). Then load your clip: "look left" pans left, "look up" tips up,
"lean in" pitches the base forward.
