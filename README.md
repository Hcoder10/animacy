# animacy

**The open interaction layer for expressive robots: human motion in, any robot's motion out — one `ROBOT.md` per body, no retraining.**

[![ci](https://github.com/Hcoder10/animacy/actions/workflows/ci.yml/badge.svg)](https://github.com/Hcoder10/animacy/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![demo](https://img.shields.io/badge/demo-live%20in%20your%20browser-brightgreen)](https://hcoder10.github.io/animacy/web/)

**▶ Live demo: <https://hcoder10.github.io/animacy/web/>** — the Autonomous Lamp and the Reachy Mini side by side, playing their vendors' own clips, human motion retargeted through each robot's `ROBOT.md`, your webcam live, and a talk mode where the robot speaks (TTS in the page) and moves in sync. No install, no build step.

<p align="center">
  <video src="https://github.com/Hcoder10/animacy/raw/main/docs/media/animacy_demo.mp4"
         poster="https://github.com/Hcoder10/animacy/raw/main/docs/media/viewer_talk_mode.png"
         controls muted playsinline width="860"></video>
</p>

<p align="center">
  <a href="https://github.com/Hcoder10/animacy/raw/main/docs/media/animacy_demo.mp4">
    <img src="docs/media/viewer_talk_mode.png" alt="animacy web viewer: talk mode, the Lamp and the Reachy Mini moving to the same speech" width="860">
  </a>
  <br>
  <em>Demo video, 3 min (click to play) · <a href="https://hcoder10.github.io/animacy/web/">open the live viewer</a></em>
</p>

## What it is, in six lines

1. **One canonical human motion space** ([`docs/CANONICAL.md`](docs/CANONICAL.md)): 28 channels at 30 Hz — head 6-DoF, gaze, brows, mouth, torso, a puppet arm, a speaking flag. Capture writes it, models predict it, robots never see anything else.
2. **One `ROBOT.md` per robot** ([`docs/ROBOT_MD_SPEC.md`](docs/ROBOT_MD_SPEC.md)): joints, limits, rest pose, safety ceiling and a linear mapping from canonical channels — signs are fixed there, never in data. `animacy check` validates it.
3. **Capture from any video or webcam** (`animacy capture`, MediaPipe face + pose + VAD), or record in the browser and import.
4. **A browser demo on the real URDFs** (three.js + urdf-loader; the JS retargeter equals the Python one to 1e-6).
5. **Speech-driven motion** ([`docs/MODEL.md`](docs/MODEL.md)): a VQ tokenizer + an autoregressive audio→codes transformer, and a retrieval (motion-matching) baseline that ships as the default; both run in the page.
6. **Runs on the real Reachy Mini today** (hardware-verified 2026-08-26) and writes exactly the CSV the Autonomous Lamp's `/servo/upload` accepts (validated against a mirror of that route's checks; **not yet run on a Lamp**).

## 60-second quickstart

Every command below was run end to end on 2026-08-27; the outputs quoted are real.

```bash
git clone https://github.com/Hcoder10/animacy && cd animacy
pip install -e ".[capture,urdf]"    # Python >= 3.10
# the bare `pip install -e .` is enough for `check` and `retarget`; [capture] adds
# MediaPipe/OpenCV/soundfile (capture, mirror, say), [urdf] adds the URDF limit
# test and `animacy preview`, [robot] adds requests for talking to a real robot

animacy check robots/lamp           # -> OK lamp: 5 joints, modes ['default', 'puppet'], urdf urdf/lamp.urdf
animacy check robots/reachy_mini    # -> OK reachy_mini: 9 joints, ...
animacy check robots/so101          # -> OK so101: 6 joints, ...

# a person talking on video -> a canonical clip (30 Hz parquet + 16 kHz wav + meta.json)
animacy capture --source path/to/talk.mp4 -o data/clips/talk --duration 60
# or your webcam:  animacy capture --source 0 -o data/clips/me --preview

# the clip -> the Lamp's own recording format (speed-legal, 30 Hz)
animacy retarget --robot lamp --clip data/clips/talk -o out/talk.csv --format autonomous_os_csv
# the same clip -> a Reachy Mini move (Pollen's recorded-move JSON)
animacy retarget --robot reachy_mini --clip data/clips/talk -o out/talk.json --format pollen_move

# find your signs headlessly: renders the calibration poses through your mapping (matplotlib Agg)
animacy preview robots/lamp --out out/preview_lamp

# hear it without a robot: text -> TTS -> motion -> printed frames
animacy say "Hi, I'm animacy." --robot reachy_mini --dry-run

# the viewer (static site, no build)
python -m http.server 8000          # -> http://localhost:8000/web/
```

On a real robot (`pip install -e ".[robot]"` for `requests`; the daemon URL is in [`robots/reachy_mini/ROBOT.md`](robots/reachy_mini/ROBOT.md)):

```bash
animacy say "Hi, I'm animacy." --robot reachy_mini --url http://<reachy>:8000 --source retrieval
animacy mirror --source 0 --robot reachy_mini --sink reachy_daemon --url http://<reachy>:8000
```

`say` speaks and moves from one waveform; `mirror` drives the robot from your webcam in real time (30 Hz, latest-sample-wins, read-back logged). `animacy lerobot`, `animacy import-browser` and `animacy profile export` cover dataset export, the viewer's Record mode, and the web profile JSON — `animacy <cmd> --help` for each.

## Robots

| robot | vendor · license | joints | URDF | native clips | signs verified in sim | verified on hardware |
|---|---|---|---|---|---|---|
| `lamp` — Autonomous Lamp | Autonomous · Apache-2.0 | 5 (vendor servo names) | from the vendor CAD (`lamp.glb` armature pivots, per-part STLs), [notes](robots/lamp/urdf/README.md) | 31 vendor recordings, verbatim | yes — against the vendor's device-measured notes and all 31 clips ([previews](robots/lamp/urdf/preview/contact_sheet.png)) | **no** (no unit on hand) |
| `reachy_mini` — Reachy Mini | Pollen Robotics / Hugging Face · Apache-2.0 | 9 (head 6-DoF, body yaw, 2 antennas) | serial visualization chain over Pollen's meshes | 16 of Pollen's emotion-library moves, converted | yes | **yes, 2026-08-26** — [evidence](docs/evidence/reachy_sim2real_20260826.md) (every axis read back within a few degrees; owner confirmed all five directions) |
| `so101` — SO-101 arm | TheRobotStudio / LeRobot · Apache-2.0 | 6 | vendor URDF, mesh paths only | — | yes (FK only) | no |

A new robot is one folder: `ROBOT.md` + a URDF. CI runs `animacy check` on every one of them.

![Lamp playing a retargeted human nod next to the vendor's hand-authored nod](docs/media/viewer_ab_vendor_nod.png)

## Add your robot in one file

[`docs/ADD_A_ROBOT.md`](docs/ADD_A_ROBOT.md) is written for a person *or* a Claude Code / Codex session: copy `robots/_template`, drop in a URDF, fill the joint table from the vendor's spec (names, limits, rest, `max_speed` from the vendor's safety file), write the `default` mapping *by function* (gaze → whatever points the face, lean → base joints, brows → the body's most legible affect channel), run `animacy check` until it passes, run `animacy preview` to read your signs off rendered PNGs, fix directions with `gain: -1`. No Python.

The Lamp and Reachy profiles are the worked examples; [`robots/lamp/urdf/README.md`](robots/lamp/urdf/README.md) shows what deriving a URDF from vendor CAD looks like when the vendor ships none. [`robots/so101/ADDING_LOG.md`](robots/so101/ADDING_LOG.md) is a coding agent adding the SO-101 by following that page, timed.

## For Autonomous OS

The Lamp and the Reachy Mini are both official Autonomous OS bodies, and animacy's `ROBOT.md` deliberately mirrors theirs (same joint names, same `max_speed` source). Autonomous OS's own `docs/not-built-yet.md` is a 21-item "claim one" list; animacy takes four of them. Owner walkthrough: [`docs/AUTONOMOUS_OS.md`](docs/AUTONOMOUS_OS.md).

**(a) Cross-body moves** — *"Lamp's 23 teleop moves exported … as a Hub dataset in Pollen's emotion-library format …, so a move recorded on either body plays on both."*
animacy makes cross-body moves by construction: a move is a *human* clip, and each body plays it through its own `ROBOT.md`.

```bash
animacy retarget --robot reachy_mini --clip data/clips/<clip> -o out/<clip>.json --format pollen_move        # Pollen recorded-move JSON: {"description","time",[{"head":4x4,"antennas":[l,r],"body_yaw"}]}
animacy retarget --robot lamp        --clip data/clips/<clip> -o out/<clip>.csv  --format autonomous_os_csv  # hal/recordings CSV: timestamp,<joint>.pos
```

Both bodies already share one dataset on the Hub: [`squaredcuber/animacy-lamp-lerobot`](https://huggingface.co/datasets/squaredcuber/animacy-lamp-lerobot) and [`squaredcuber/animacy-reachy-mini-lerobot`](https://huggingface.co/datasets/squaredcuber/animacy-reachy-mini-lerobot) are the *same* human episodes retargeted to each robot, LeRobot v3.0, with the speech features alongside ([`docs/LEROBOT.md`](docs/LEROBOT.md)). Pollen's own library also imports: `scripts/pollen_npz_to_joints.py` converts [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) moves into animacy joint tables (that is what plays as "native" Reachy clips in the viewer). What animacy does **not** do: turn a robot-authored move back into human motion — a Pollen move does not become a Lamp move.

**(b) Community moves** — *"Community moves on Reachy: … a `HAL_REACHY_MOVES` list so any `reachy_mini_community_moves` dataset — a move you recorded and pushed to the Hub — plays by name…"*
`animacy capture` (webcam, phone video, or the viewer's Record mode + `animacy import-browser`) → `animacy retarget --format pollen_move` writes the file format their driver loads. Pushing it as a `reachy_mini_community_moves` dataset is a plain `huggingface_hub` upload; animacy's own pushers (`scripts/push_hf.py`, `scripts/export_lerobot.py --push`) publish the human corpus and the LeRobot datasets and refuse any clip without a license record.

**(c) Recorded animations under the safety gate** — *"Recorded animations under the safety gate. `motion.max_speed` is enforced on the commanded paths … but not on the one a body moves by most: `_continue_playback` replays stored frames at a fixed fps and reads no bound."*
animacy makes the design call their note describes: **stretch time, never clip or drop.** `retarget_clip` widens only the frame gaps that would exceed `max_speed` (the same rule as their `recording_timing.stretch_timeline`), then a causal `rate_limit` guarantees legality exactly — 0 speed-cap violations on every clip in [`docs/RESULTS.md`](docs/RESULTS.md). `animacy.export.validate_autonomous_os_csv` mirrors `hal/routes/servo.py:upload_servo_recording` plus a per-joint speed check, so a file that passes here is accepted there. Live streaming (`/servo/move`) is held to `SAFETY.md motion.max_speed = 120°/s` by their gate; the profile records both ceilings ([`robots/lamp/ROBOT.md`](robots/lamp/ROBOT.md)).

**(d) A live policy** — *"A live policy behind the marker — the interface now exists: `POST /policy/run` accepts `{"policy":…,"task":…}` and records it as a **dry run** only … Still needed: local or async LeRobot inference, an SO-101 motion driver and calibrated joint map, a safety-gated target loop…"*
Their endpoint is a dry-run recorder today (`{"policy","task"}` → `state: "dry_run"`). animacy's live loop is the executor shape it needs: `animacy say "<text>" --robot lamp --sink autonomous_os_hal --url http://<lamp>:5001` turns text into speech, speech into canonical motion (retrieval or the learned model), motion into `/servo/move` frames at 30 Hz through the Lamp's `ROBOT.md` and under their speed gate, while the audio plays. `docs/AUTONOMOUS_OS.md` sketches the `PolicyService` adapter. Their item also asks for "an SO-101 motion driver and calibrated joint map"; [`robots/so101/`](robots/so101/) is exactly that shape — a `ROBOT.md` with the vendor joint names, limits and rest pose plus the vendor URDF — though animacy drives it only in simulation and has never commanded an SO-101. **None of (d) has been run on a Lamp.**

## Results

Every number here is copied from [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/evidence/`](docs/evidence/), which in turn cite a `checkpoints/<run>/REPORT.md` with the exact training command. Held-out = a speaker the model never saw.

**Data.** The corpus is **73 license-verified clips, 320.8 valid minutes, 37 speakers** (98 captured, 25 dropped by the face-valid/duration gate; counts recomputed from `data/clips/_index.json`), published as [`squaredcuber/animacy-human-motion`](https://huggingface.co/datasets/squaredcuber/animacy-human-motion). A continuous harvest toward a much larger corpus runs alongside it ([`docs/HARVEST.md`](docs/HARVEST.md)).

**Tokenizer** (VQ-VAE, 512 × 64, one code per 2 frames, v2a): 509/512 codes used, val MAE 0.23 vs 0.62 for predict-the-mean. Dead-code revival plus data-dependent initialisation were required — the stock EMA quantiser collapsed to one code at this data size.

**Audio → motion, two held-out speakers** (v2a autoregressive decoder):

| held-out speaker | code NLL (unigram floor) | head-beat recall vs shuffled audio | stillness (truth) | velocity W1 rel. |
|---|---|---|---|---|
| obama_2015 | **3.557** (5.663) | 0.438 / 0.497 (−0.06) | 0.084 (0.103) | **0.19** |
| kende | **3.603** (5.481) | 0.391 / 0.403 (−0.01) | 0.090 (0.097) | 1.07 |

The AR model predicts unseen speakers' motion codes ~2 nats below the unigram floor and matches human stillness to within 0.02 — it neither jitters nor freezes. What it does **not** show is beat alignment: on neither speaker does any learned generator beat its own shuffled-audio control by the required 0.05 head-beat recall. **So we make no claim that the learned motion is timed to the speech, and retrieval — real human windows matched to the speech, aligned by construction — ships as the default source** (`web/models/model.json: default_backend = retrieval`). The AR model is exported and selectable. The promotion rule lives in the trainer so a future run flips the default only on evidence.

**Retarget legality:** 0 speed-cap violations for every source on both robots.

**Sim-to-real:** Reachy Mini, physical unit, canonical clip → `ROBOT.md` → daemon at 30 Hz; every commanded axis read back within a few degrees, owner confirmed directions; `animacy say` ran on the robot with audio in sync ([evidence](docs/evidence/reachy_sim2real_20260826.md)).

**Browser:** JS retargeter = Python to 1e-6 on 240 random frames × 2 robots × 2 modes; 240 fps on an RTX 5080 laptop, 8–12 fps under software rendering.

**Tests:** `python -m pytest -q` on a clean checkout — **213 passed, 5 skipped** (218 collected; the 7 Autonomous-OS HAL integration tests need a live HAL and collapse to a single module-level skip without one, giving 224 collected when a HAL *is* running). See *Status* below for the one test that fails against that simulator.

### The blind-grader gate — currently FAILING, on purpose reported

An outside judge (Kimi K3 through the local `kimi` CLI) watches reels of short robot clips rendered through the same viewer, blind: the clip→origin map is sealed in a manifest the judge never sees, and the rubric carries no project vocabulary. The pass rule is owned by the gate ([`animacy/grade`](animacy/grade), spec in [`docs/GRADING.md`](docs/GRADING.md)) and may not be weakened elsewhere: for each robot, the shipped default source must score **overall ≥ 8.0 on all five movements**, using the mean over seeds (best-of-seeds is reported, never used).

The same judge also grades **the vendors' own hand-authored clips**, blind, in the same reel — so every score has a reference point made by the people who built the robot. Latest run — **run 3, 2026-08-27** ([full report](docs/evidence/grading/20260827_1501_run3.md)), source under test `retrieval` (the shipped default), on the sealed held-out lines:

**Lamp** (the judge's vendor calibration passes here: vendor mean 6.6, minimum 6.0 — OK)

| movement | animacy | the vendor's own clip | |
|---|---|---|---|
| greeting | 6.0 | 6.0 | level |
| agreement | 5.0 | 6.0 | below |
| doubt | 6.0 | 7.0 | below |
| excitement | **7.0** | 6.0 | **above** |
| thinking | 6.0 | 8.0 | below |
| **mean** | **6.0** | **6.6** | **below** |

So on the Lamp, animacy's generated motion is level with or better than Autonomous's hand-made clips on 2 of 5 movements and behind on 3, ending 0.6 points below overall. **The 8.0 bar we set ourselves is not met by anything — not by animacy, and not by the vendor's own clips either.** Verdict: **FAIL** (minimum movement 5.0). Best-of-seeds would still fail.

**Reachy Mini** scored 6.0 / 5.0 / 6.0 / 7.0 / 7.0 (mean 6.2) against a vendor mean of 5.6 — but **we do not claim that win**: the gate's own calibration check flags that run as `BROKEN — vendor clips average below the minimum: the rendering or the rubric is broken, candidate scores are not trustworthy`. Verdict there is **FAIL** as well, and the harness's warning is printed in the evidence file rather than dropped.

Judge self-consistency on identical clips is ±0.8 (lamp) / ±0.6 (reachy). The tuning-minus-held-out gap is +0.0 (lamp) / +0.2 (reachy) — no evidence the motion was tuned to the known lines. All three runs are recorded in full, including the two earlier ones that also failed: [run 1](docs/evidence/grading/20260826_2320.md), [run 2](docs/evidence/grading/20260827_0301_run2.md), [run 3](docs/evidence/grading/20260827_1501_run3.md). No claim of passing appears anywhere in this repo.

## Data and licence policy

- **Human corpus:** [`squaredcuber/animacy-human-motion`](https://huggingface.co/datasets/squaredcuber/animacy-human-motion) — canonical clips (`motion.parquet`, `audio.wav`, `meta.json`) with sources and licenses listed verbatim in the card.
- **Licence policy** (`scripts/fetch_sources.py`, `scripts/push_hf.py`): public-domain and CC-BY talking-head video only, license verified from metadata, ND refused, evidence copied into every clip's `meta.json`; nothing without a license record is pushed. Your own webcam clips carry `license: self` (CLI) or CC-BY-4.0 (browser Record mode).
- **Robot-space datasets** for imitation learning (`animacy lerobot --robot <name> --out <dir> --validate --push <repo>`): the two LeRobot v3.0 datasets above (`observation.state`/`action` in each robot's units, 66-d speech features + speaking flag as `observation.environment_state`, validated with the real `LeRobotDataset` loader).
- The vendors' hand-authored clips (Lamp 31, Pollen's Reachy library) are **not** training data; they are the envelope each retarget is tuned to, the A/B in the viewer, and the calibration set for the blind grader.

## Architecture

![architecture](docs/media/architecture.svg)

```
animacy/schema.py      the canonical frame (CHANNELS is the single source of truth)
animacy/capture.py     video/webcam -> canonical clips        animacy/mirror.py    live webcam -> robot
animacy/profile.py     ROBOT.md parser + `animacy check`      animacy/retarget.py  the mapping core (== web/js/retarget.js)
animacy/preview.py     headless calibration-pose renders      animacy/export.py    Lamp CSV, Pollen move JSON
animacy/features.py    audio features (== web/js/features.js) animacy/sinks.py     Reachy daemon, Autonomous HAL
animacy/model/         vq, a2m_ar, retrieval, train, ONNX     animacy/serve.py     `animacy say`: text -> TTS -> motion -> robot
animacy/grade/         the blind acceptance gate              web/                 the viewer (static, no build)
robots/<name>/         ROBOT.md + urdf/ + meshes/ + clips/    docs/                the contracts and the evidence
```

Full documentation index: [`docs/README.md`](docs/README.md). Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Status — what is done, what is not

**Done and verified:** the canonical schema and `ROBOT.md` contract with `animacy check`; capture from video/webcam/browser; retargeting with an exact speed-cap guarantee; Lamp, Reachy Mini and SO-101 URDFs; Reachy Mini sim-to-real on a physical unit; the browser viewer with talk mode (Kokoro TTS in the page); two LeRobot v3.0 datasets validated with the real loader; 213 passing tests.

**Not done, or not verified — read before quoting anything above:**

- **No Lamp hardware yet.** The Lamp URDF, signs, CSV export and HAL sink are verified against the vendor's CAD, device notes, recordings and route code, not on a unit. The vendor's joint values are calibration-span units (`RANGE_M100_100`), treated as degrees (~1.07°/unit yaw, ~1.16°/unit roll, pitch joints calibration-dependent) — directions and topology are exact, **amplitudes approximate** ([`robots/lamp/urdf/README.md`](robots/lamp/urdf/README.md)).
- **One test fails against the vendor's laptop HAL simulator.** `tests/test_lamp_hal.py::test_playback_tracks_the_uploaded_frames` asserts the simulator's reported joint positions track the uploaded CSV within 2°; observed max error is 12.8°, and 7.9° at the best-fit time offset, so it is not merely a clock skew. The upload/accept and the five reject-parity tests against the same simulator pass, so the *file* is right; how faithfully that simulator replays it is unresolved. The test is skipped unless a HAL answers at `LAMP_HAL_URL`, so a clean checkout and CI report 213 passed / 5 skipped.
- **The blind-grader gate fails** (above). The learned model does not beat its shuffled-audio control on beat timing on either held-out speaker; retrieval is the default.
- **Listen mode** (microphone → causal model) is experimental. Reachy antenna out/in geometry has not been eyeballed on hardware (the mapping is documented either way). SO-101 is sim only.
- A robot-authored move cannot be moved to another body (**no inverse retarget**).
- The corpus index records no language field, so no language count is claimed here even though the sources are visibly multilingual.

## License and attributions

- animacy: **Apache-2.0** ([`LICENSE`](LICENSE)) for code, docs and profiles, except where a folder carries its own license file.
- Autonomous Lamp CAD, recordings and joint conventions: [Autonomous OS](https://github.com/autonomous-ai/autonomous-os) `robots/` tree, Apache-2.0 ([`robots/lamp/meshes/ATTRIBUTION.md`](robots/lamp/meshes/ATTRIBUTION.md)). Their `hal/` is GPL-3.0 and is **never vendored** — it is cloned into a gitignored `third_party/` only to run evidence checks against.
- Reachy Mini description, meshes and emotion library: Pollen Robotics / Hugging Face, Apache-2.0 ([`robots/reachy_mini/meshes/ATTRIBUTION.md`](robots/reachy_mini/meshes/ATTRIBUTION.md)).
- SO-101: TheRobotStudio SO-ARM100, Apache-2.0 ([`robots/so101/meshes/ATTRIBUTION.md`](robots/so101/meshes/ATTRIBUTION.md)).
- LeLamp (GPL-3.0) was read as a kinematic reference for the Lamp's ancestry only; **no LeLamp file or mesh is in this repository.**
- Video sources: public domain and CC-BY only, credited per clip in the dataset card; captured motion of the author under `license: self`.

Lineage: [reachy-duplex](https://github.com/Hcoder10/reachy-duplex) (full-duplex speech + learned motion on a physical Reachy Mini). Built for the Autonomous Open Source Grant, August 2026.
