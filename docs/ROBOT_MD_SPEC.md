# `ROBOT.md` — one file adds a robot (`animacy.robot.v1`)

A robot joins animacy by dropping `robots/<name>/ROBOT.md` next to a URDF. The
file is YAML front matter (machine-read) followed by prose (human- and
agent-read). Nothing else is required: no Python, no retraining. `animacy check
robots/<name>` validates it; `animacy retarget --robot <name>` uses it; the web
viewer loads the JSON that `animacy profile export` writes from it.

This mirrors Autonomous OS's own `ROBOT.md` convention on purpose — a body
declared there can be declared here with the same joint names.

## Front matter

```yaml
---
schema: animacy.robot.v1
name: lamp                       # slug, [a-z0-9_]
display_name: Autonomous Lamp
vendor: Autonomous
homepage: https://github.com/autonomous-ai/autonomous-os
license: Apache-2.0              # of the description + meshes you ship here
rate_hz: 30                      # rate the robot consumes frames at

description:
  urdf: urdf/lamp.urdf           # relative to this file
  mesh_scale: 1.0
  up_axis: z
  viewer:                        # hints for the web viewer only
    camera_distance: 0.9
    ground: true

joints:                          # the robot's controllable joints, in servo order
  - name: base_yaw               # name used in every animacy file for this robot
    unit: deg                    # deg | mm
    min: -90
    max: 90
    rest: 0                      # value at the robot's neutral/idle pose
    max_speed: 120               # unit/s, hard ceiling (from the vendor's safety file)
    urdf_joint: base_yaw         # joint name inside the URDF (default: same as name)
    urdf_sign: 1                 # +1/-1: animacy value * sign (+ offset) = URDF joint value
    urdf_offset: 0               # in `unit`
  # ...

retarget:                        # one or more named mappings from canonical channels
  default:                       # `default` is required; others are selectable
    base_yaw:
      from: head_yaw             # a channel from docs/CANONICAL.md
      gain: 1.0
    base_pitch:
      mix:                       # linear mix of several channels
        - { from: torso_lean_fwd, gain: 1.2 }
        - { from: head_x, gain: 0.15 }
      offset: 0                  # added after the mix, in joint units, relative to `rest`
      min: -40                   # optional, tighter than the joint limit
      max: 60
      deadband: 0.5              # |value| below this → 0 (kills tracker jitter)
      smooth_hz: 6               # one-pole low-pass cutoff for live use; offline uses zero-phase
      # v1.1 keys — all optional, absent = v1 behaviour (exact equations: docs/RETARGET.md)
      spring: { hz: 4, zeta: 0.7 }   # 2nd-order tracker instead of smooth_hz (overshoot-and-settle)
      idle: { amp: 2, hz: 0.2 }      # deterministic sway added while the target is still; `still:` optional
      soft_limit: 0.15               # tanh knee over the last 15 % of the range before the hard clamp
  puppet:                        # e.g. drive the lamp with your own arm
    base_yaw:      { from: shoulder_yaw }
    base_pitch:    { from: shoulder_pitch, gain: -1, offset: 90 }
    # ...

export:
  formats: [autonomous_os_csv, lerobot]
  autonomous_os_csv:
    column_suffix: ".pos"        # base_yaw -> base_yaw.pos
    timestamp_column: timestamp
    fps: 30

runtime:                         # how `animacy serve` reaches the real body
  kind: autonomous_os_hal        # autonomous_os_hal | reachy_sdk | http_json | none
  url: http://127.0.0.1:5001
  stream_hz: 30

native_clips:                    # the vendor's own hand-authored moves, if any
  dir: clips/native
  format: autonomous_os_csv
---
```

### Rules

1. **Joint names are the contract.** Every animacy file for this robot uses
   `joints[].name`; the URDF may differ (`urdf_joint`, `urdf_sign`,
   `urdf_offset`) so vendor URDFs are used unmodified.
2. **Values are `rest`-relative in `retarget`.** `joint = rest + offset + Σ gain·channel`,
   then deadband, then (if `soft_limit`) a tanh knee, then clamped to
   `[min, max]` (mapping bounds if given, else joint bounds), then (if `idle`)
   the gated sway is added, then tracked (`spring` if given, else the
   `smooth_hz` one-pole; offline uses a zero-phase filter for the one-pole
   case), then rate-limited to `max_speed` (offline: time is *stretched* so
   nothing is clipped; live: velocity is clipped — always the last step
   before the hard clamp). `docs/RETARGET.md` is the exact per-frame spec.
3. **Every `from` must be a canonical channel** (`docs/CANONICAL.md`). Unknown
   channel = validation error, not silently zero.
4. **Signs are fixed here, never in captured data.** If "look up" moves the
   robot down, set `gain: -1` on that joint.
5. **`max_speed` comes from the vendor's safety file** when one exists
   (Autonomous Lamp: `SAFETY.md motion.max_speed: 120` deg/s). Do not raise it
   because the servo can go faster.
6. `animacy check` must pass before a robot is merged. It verifies: unique joint
   names; the URDF exists and contains every `urdf_joint`; rest within limits;
   every mapping channel exists; `default` mapping present; each joint mapped at
   most once per mode; speeds positive.
7. **`max_speed` when the vendor publishes no safety file** (LeRobot arms, most
   research robots): use the *lowest* of (a) the servo datasheet's no-load speed
   × 0.5, (b) the fastest move in the vendor's own recordings/datasets, (c)
   180 deg/s (100 mm/s) — and say which in the prose. Conservative beats fast:
   time is stretched, never clipped, so nothing is lost.
8. **`rest` when the vendor has no idle/emotion library**: a natural *attentive*
   pose facing +x (the "listening" pose), not a storage/folded pose. State it in
   the prose with the joint values.
9. **Units promise.** `export.formats: [lerobot]` profiles promise degrees /
   millimetres in the LeRobot dataset (`use_degrees=True` semantics). Vendors
   that drive servos in calibrated-span units (Autonomous OS `RANGE_M100_100`)
   keep vendor units in the profile and say so in a `# UNITS CAVEAT` comment.
10. **Profile limits must lie inside the URDF's limits.** `animacy check`
    converts `min/max` through `urdf_sign`/`urdf_offset`/unit and refuses a
    range wider than the URDF's `<limit>` (1e-3 tolerance) — a wider range means
    the sign or offset is wrong, not the URDF.

Valid `export.formats` today: `autonomous_os_csv`, `pollen_move`, `lerobot`,
`csv`, `json`. Planned (not yet implemented): an `absolute: true` mapping flag so
1:1 puppet chains need no `offset: -rest` per joint.

## Prose section (after the front matter)

Write for a person *or an agent* who has never seen the robot:

- **Sign conventions, verified on hardware.** e.g. "positive `base_yaw` turns the
  head toward the robot's left (checked on unit lamp-0c89)". If unverified, say
  so — a wrong sign drives the head away from a face in a runaway.
- **Neutral pose** and what `rest` looks like physically.
- **What each channel should *mean* on this body** (brows → antennas because
  both are the owner's affect channel; head_x → lean).
- **How to get frames onto the real robot** (endpoint, format, rate, safety
  ceiling).
- **Verification checklist** — commands to run, what you should see.

## Adding a robot with Claude Code / Codex

Give the agent `docs/ADD_A_ROBOT.md`. In short: copy `robots/_template/`, put
the URDF + meshes in, fill the joint table from the vendor's spec, write the
`default` mapping by *function* (gaze → yaw/pitch joints, lean → base joints,
affect → whatever the body's most legible channel is), run
`animacy check` until it passes, then `animacy preview --robot <name>
--clip <clip>` to eyeball it in the browser, and fix signs with `gain: -1`.
