# The canonical human motion space (`animacy.human.v1`)

Everything in animacy flows through one representation: a **human** doing the
expressive thing, sampled at a fixed rate. Capture writes it, the motion model
predicts it, and every robot consumes it through its own `ROBOT.md` mapping.
A robot never sees raw landmarks and a model never sees a servo — that is what
lets one model drive a lamp, a Reachy Mini, and whatever you add tomorrow.

## Frame

One row per tick at `rate_hz` (default **30 Hz**). All values are floats,
**relative to a neutral pose** captured at the start of a session (looking
straight at the camera, relaxed). Missing channels are `NaN`, and each optional
group carries a validity flag so consumers can mask instead of guess.

| channel | unit | range | meaning |
|---|---|---|---|
| `t` | s | ≥0 | time since clip start |
| `head_yaw` | deg | ±90 | **+ = turn LEFT** (subject's left) |
| `head_pitch` | deg | ±60 | **+ = look UP** |
| `head_roll` | deg | ±45 | **+ = tilt so the subject's RIGHT ear drops toward their right shoulder** |
| `head_x` | mm | ±150 | head translation, **+ = forward** (lean in toward the camera) |
| `head_y` | mm | ±150 | **+ = subject's left** |
| `head_z` | mm | ±150 | **+ = up** |
| `gaze_yaw` | deg | ±40 | eye direction relative to the head, + = left |
| `gaze_pitch` | deg | ±30 | + = up |
| `brow_l`, `brow_r` | 0..1 | | eyebrow raise (0 = neutral, 1 = max raise) |
| `brow_furrow` | 0..1 | | brows pulled down/in |
| `eye_open_l`, `eye_open_r` | 0..1 | | 0 = closed, 1 = wide open (neutral ≈ 0.6) |
| `mouth_open` | 0..1 | | jaw open |
| `smile` | 0..1 | | mouth corners up |
| `torso_lean_fwd` | deg | ±45 | + = leaning toward the camera |
| `torso_lean_side` | deg | ±45 | + = leaning to the subject's left |
| `torso_yaw` | deg | ±90 | + = shoulders turn left |
| `arm_valid` | 0/1 | | 1 when the puppet arm below was tracked this frame |
| `shoulder_yaw` | deg | ±90 | + = upper arm swings left (horizontal) |
| `shoulder_pitch` | deg | -30..180 | 0 = arm hanging down, 90 = horizontal forward, 180 = straight up |
| `elbow_flex` | deg | 0..150 | 0 = straight, + = bending |
| `wrist_roll` | deg | ±90 | forearm pronation, + = thumb rotates left |
| `wrist_pitch` | deg | ±80 | + = hand tips up |
| `hand_open` | 0..1 | | 0 = fist, 1 = spread |
| `speaking` | 0/1 | | subject is talking this frame (from VAD on the clip's audio) |
| `face_valid` | 0/1 | | face channels tracked this frame |

Coordinate frame: right-handed, **x forward, y left, z up** (ROS body frame),
viewed from the subject. Signs above are the contract; a `ROBOT.md` fixes any
robot-side convention with `gain: -1`, never by editing captured data.

The puppet arm defaults to the subject's **right** arm. `capture --arm left`
mirrors it into the same channels so downstream never cares.

## Why these channels

They are the channels that carry *animacy* in conversation and that at least
one target body can render:

- head 6-DoF + brows: what a talking human moves most; maps 1:1 onto a
  Stewart-platform head (Reachy Mini) and onto a lamp's neck/head.
- brows → antennas / lamp head-tip: each is its owner's most legible affect
  channel (function-preserving retarget, not geometry).
- torso lean: the lamp's `base_pitch`/`elbow_pitch` are exactly "lean in / sit back".
- the arm chain (shoulder yaw/pitch, elbow, wrist roll/pitch): a 5-DoF desk
  lamp **is** an arm — puppeteering it with your own arm is a near-identity map.
- `speaking`: a listener and a speaker move differently; training without the
  flag teaches listening-as-speaking (measured on Reachy work — see reachy-duplex).

## Files

A clip is a directory:

```
<clip>/
  motion.parquet     # the frame table above (pyarrow), metadata in the schema
  audio.wav          # 16 kHz mono, same clock as motion.t (optional)
  meta.json          # source, license, capture settings, neutral pose, notes
```

`meta.json` keys: `source` (webcam|video|file), `source_url`, `license`
(SPDX or "self"), `rate_hz`, `subject` (opaque id), `arm` (left|right|none),
`neutral` (raw neutral pose used for zeroing), `tool_versions`.

Python: `animacy.schema.HumanClip` reads/writes this; `CHANNELS` is the ordered
column list and the single source of truth for every consumer (retarget, model,
web JSON export).

## Where this is used

- [`ROBOT_MD_SPEC.md`](ROBOT_MD_SPEC.md) — the other contract: how a robot
  declares its mapping *from* these channels. [`ADD_A_ROBOT.md`](ADD_A_ROBOT.md)
  is the walkthrough.
- [`RETARGET.md`](RETARGET.md) — how a channel becomes a joint value, and the
  speed-cap guarantee.
- [`MODEL.md`](MODEL.md) — what predicts these channels from speech.
- [`HARVEST.md`](HARVEST.md) — how clips in this format are collected at scale.
- [`LEROBOT.md`](LEROBOT.md) — how they are exported for imitation learning.
- Docs index: [`README.md`](README.md).
