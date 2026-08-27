# animacy on an Autonomous Lamp (Autonomous OS)

A walkthrough for a Lamp owner: turn a human clip into a move the Lamp plays,
put it on the unit, trigger it from the agent, and (optionally) stream live.
Everything here targets HAL's HTTP API as it is in
[autonomous-os](https://github.com/autonomous-ai/autonomous-os) `hal/routes/servo.py`
and `hal/models.py`. **None of it has been run on a physical Lamp yet** — the
request shapes and the CSV checks are taken from the vendor's code, and the
URDF/sign work is verified in simulation only (`robots/lamp/urdf/README.md`).
If you own a unit and run this, please open an issue with what you saw.

## What the Lamp expects

- Joints, in servo order: `base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch`
  (`hal/models.py ServoMoveRequest`; API-clamped to ±90).
- A recording is a CSV with a `timestamp` column (seconds) and `<joint>.pos`
  columns — `hal/recordings/*.csv`, 20 Hz as shipped; HAL resamples to its
  30 Hz loop on load and stretches any segment faster than 250°/s
  (`hal/drivers/motors/recording_timing.py`).
- Commanded moves (`POST /servo/move`) go through the safety gate:
  `SAFETY.md motion.max_speed: 120` °/s, "the servo route stretches a move's
  duration so no joint exceeds it" (`hal/safety/policy.py`).
- The agent moves the body by emitting markers in its reply, e.g.
  `[HW:/servo/play:{"recording":"nod"}]` (`hal/drivers/voice/voice_service.py`
  parses `[HW:/path:{json}]`).
- Values are the vendor's calibration-span units, not literal degrees
  (`hal/follower/config_hal_follower.py use_degrees = False`, LeRobot
  `RANGE_M100_100`). animacy treats them as degrees; see the caveat in
  `robots/lamp/ROBOT.md`.

`robots/lamp/ROBOT.md` encodes all of this: joint names, ±90 limits, `rest`
= the vendor's idle median, `max_speed: 250` for recordings and
`runtime.extra.stream_max_speed: 120` for live commands, the routes.

## Step 1 — get a canonical clip

Any of:

```bash
animacy capture --source path/to/person_talking.mp4 -o data/clips/talk --duration 60   # a video file
animacy capture --source 0 -o data/clips/me --preview                                   # your webcam, q to stop
animacy import-browser recording.zip -o data/clips/browser                              # the viewer's Record mode
```

or download one from the human corpus
([`squaredcuber/animacy-human-motion`](https://huggingface.co/datasets/squaredcuber/animacy-human-motion)):
`clips/<name>/motion.parquet` is a clip directory as-is.

## Step 2 — retarget to the Lamp

```bash
animacy retarget --robot lamp --clip data/clips/talk -o out/talk.csv --format autonomous_os_csv
# -> wrote out/talk.csv (N frames, T s, mode=default, format=autonomous_os_csv)
```

What the file is: `timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,wrist_roll.pos,wrist_pitch.pos`
at 30 Hz, values in the Lamp's own units, produced by `retarget_clip`:
`rest + Σ gain·channel` per joint from the `default` mapping → deadband →
clamp to the mapping bounds → **time stretched** wherever a joint would exceed
`max_speed` (250°/s; only the impossible segments grow, like the vendor's own
`stretch_timeline`) → resampled to 30 Hz → zero-phase smoothed → a causal
`rate_limit` that guarantees the ceiling exactly. Nothing is clipped, so a big
human move becomes a slower Lamp move, never a truncated one.

`--mode puppet` maps your own arm instead of your face (shoulder → base,
elbow → elbow, wrist → neck/head).

## Step 3 — validate before uploading

`animacy.export.validate_autonomous_os_csv` is a mirror of
`hal/routes/servo.py:upload_servo_recording` (`timestamp` column present,
`<name>.pos` columns only, known joints, numeric values, row/size caps) plus a
per-joint speed check. An empty list means the Lamp would accept the file:

```python
from animacy.export import validate_autonomous_os_csv
from animacy.profile import load_profile

prof = load_profile("robots/lamp")
errs = validate_autonomous_os_csv(
    open("out/talk.csv", encoding="utf-8").read(),
    valid_joints=prof.joint_names,
    max_speed={j.name: j.max_speed for j in prof.joints},
)
print(errs or "OK")
```

## Step 4 — upload and play

```bash
LAMP=http://<lamp-ip>:5001            # HAL; on the unit itself http://127.0.0.1:5001
curl -sX POST $LAMP/servo/upload -F "file=@out/talk.csv" -F "recording_name=talk"
curl -sX POST $LAMP/servo/play   -H 'Content-Type: application/json' -d '{"recording":"talk"}'
curl -sX POST $LAMP/servo/stop
```

Notes from the route code: the name is sanitised to letters, digits, `_`, `-`
(max 64 chars); uploads are capped at 2 MB and 20 000 rows (a 30 Hz clip of a
few minutes is well inside); `/servo/play` is silently ignored while the
device is asleep or in suppressed mode (it returns `{"status":"ok"}` either
way); `/servo/hold` and `/servo/release` freeze / de-torque the body;
`GET /servo` reports the servo state. HAL that is reached over the LAN
rather than from the board sits behind its local-only / bearer gate
(`hal/server.py`) — add whatever credential your unit's setup gave you.

## Step 5 — trigger it from the agent

Once uploaded, the move is a named recording like any of the 31 the Lamp
ships with, so the agent plays it the same way it plays `nod`:

```
[HW:/servo/play:{"recording":"talk"}]
```

Put that marker in a skill (below) or in the character's instructions, one
marker per clause; the Emotion skill's own markers (`[HW:/emotion:{...}]`) can
precede it for the LED side.

## Live path — `animacy say` and `animacy mirror` on the HAL sink

`animacy.sinks.AutonomousHalSink` streams `POST /servo/move` frames:

```bash
pip install requests
animacy say "Hey! I'm animacy." --robot lamp --sink autonomous_os_hal --url $LAMP --source retrieval --checkpoint checkpoints/v1
animacy mirror --source 0 --robot lamp --sink autonomous_os_hal --url $LAMP --preview
```

- `say`: text → TTS waveform (SAPI on Windows, `espeak-ng` elsewhere,
  `kokoro-onnx` optional) → audio features → motion source (`retrieval`,
  `model`, or the labelled `envelope` heuristic when no checkpoint is around)
  → canonical frames → `retarget_clip` → `stream_table` at 30 Hz with a slew
  clamp, while the audio plays locally. Motion and speech come from the same
  samples, so sync is structural.
- `mirror`: webcam → MediaPipe → canonical channels → `LiveRetargeter` → sink,
  30 Hz sender clock, latest-sample-wins.
- Each frame is `{"positions": {"<joint>.pos": v, ...}, "duration": 0.05}`.
  HAL's gate stretches the duration of any frame that would exceed
  `SAFETY.md motion.max_speed` (120°/s), so the live path can never be faster
  than the vendor allows; the profile's `stream_max_speed: 120` documents it.
- The sink's `neutral()` plays the vendor's `idle`; `stop()` posts `/servo/stop`.

Status: **unverified on hardware.** The sink is built from `hal/models.py`
and `hal/routes/servo.py`; the first thing to check on a real unit is that a
0.05 s `duration` is accepted at 30 Hz without the gate queueing up (if it
does, raise `live_duration` in `AutonomousHalSink` or lower `stream_hz`).

## A skill the agent can use: `skills/animacy/SKILL.md`

Autonomous OS skills are one folder with a `SKILL.md` (front matter `name`,
`description`, then steps; hardware via `[HW:/path:{json}]` markers, optional
`skill.json` capability list). A minimal one that lets the agent "express
&lt;description&gt; with a generated move":

```markdown
---
name: animacy
description: "Express a feeling or a reaction with a generated body move (animacy).
  Use when the user asks the lamp to react, emote, mirror them, or move to what it
  is saying, and no built-in emotion fits: 'lean in', 'think about it', 'do a slow
  nod', 'react like you're surprised', 'move while you talk'."
---
# animacy moves

Moves generated from human motion are uploaded to this unit as recordings named
`animacy_<name>`. Pick the closest one and play it; say nothing about the mechanism.

1. Match the request to a recording from this catalogue (animacy names carry
   their intent): `animacy_nod_soft`, `animacy_nod_big`, `animacy_lean_in`,
   `animacy_sit_back`, `animacy_look_around`, `animacy_think`, `animacy_surprise`,
   `animacy_listen_still`.
2. Reply with one marker per clause, in order:
   `[HW:/servo/play:{"recording":"animacy_lean_in"}]`
   For a reaction that also needs the light, put the Emotion marker first:
   `[HW:/emotion:{"emotion":"curious","intensity":0.6}]` `[HW:/servo/play:{"recording":"animacy_think"}]`
3. If nothing in the catalogue fits and animacy is installed on the unit, generate one
   from what you are about to say and let it stream live while you speak:
   `animacy say "<your reply>" --robot lamp --sink autonomous_os_hal --url http://127.0.0.1:5001`
   (this is a shell step; it respects SAFETY.md through the servo gate).
4. Confirm briefly in words; never list recording names to the user.

Do not use these moves for tracking a person or aiming at furniture — that is
servo-tracking / `[HW:/servo/aim:...]`.
```

`skill.json`: `{"capabilities": ["motion"]}` so it installs only on bodies
with servos. The catalogue names are whatever you chose at upload
(`recording_name=`); HAL has no HTTP route that lists recordings
(`python -m hal.list_recordings` on the unit does), so keep the list in the
skill text and regenerate it when you upload more. Step 3 assumes animacy is
installed on the board (pure numpy/pandas/scipy for the core; `espeak-ng` for
TTS; a retrieval checkpoint for anything better than the envelope heuristic).

## `POST /policy/run`

Today (`hal/routes/policy.py`, `hal/policy/service.py`) the route is a
dry-run recorder: `{"policy": "lerobot/smolvla_base", "task": "pick up the mug"}`
→ `{"status":"accepted","state":"dry_run","dry_run":true,...}`; `GET /policy`
shows it; `POST /policy/stop` clears it; nothing is inferred and no motor is
touched. The comment in `service.py` names the condition for replacing it: "Do
not replace it with an inference implementation until the motion driver owns
safety-clamped target delivery and `/servo/stop` cancels its worker."

The `PolicyService` protocol it defines (`run(policy, task) -> PolicyRun`,
`stop()`, `active_run()`) is exactly the shape of animacy's talk loop:

```
run("animacy/talk", task=<text>)   -> serve.say(text, profile, source="retrieval", sink=AutonomousHalSink)
                                      in a worker thread; frames go through /servo/move (the safety gate)
stop()                             -> stop the worker, sink.stop() (= /servo/stop), then sink.neutral()
active_run()                       -> the PolicyRun with state "running" and the utterance
```

That adapter is **not written**; it is a ~50-line class against
`animacy.serve.say` and `animacy.sinks.AutonomousHalSink`, and it should not
exist until the vendor's own precondition above holds. What exists is the
executor it would call, verified end to end on the Reachy Mini
(`docs/evidence/reachy_sim2real_20260826.md`) and built to the HAL request
shapes for the Lamp.

## Checklist for the first run on a real Lamp

1. `animacy check robots/lamp` passes.
2. Play the vendor's own `nod` and `headshake` through the viewer
   (`web/`, Native clip tab) — this is what the URDF was verified against.
3. Upload and play one animacy CSV (Steps 2–4). Watch: does `idle → talk →
   idle` return home? Does the head keep looking at you rather than tucking?
   If a direction is wrong, the fix is `gain: -1` on that joint's mapping in
   `robots/lamp/ROBOT.md`, never a change to the CSV.
4. If amplitudes look off by a constant factor on a pitch joint, that is the
   units caveat (calibration-span units vs degrees); note the factor in an issue.
5. Try the live path with `--source envelope` first, then `retrieval`.
