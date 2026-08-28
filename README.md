# animacy

**Human motion in, any robot's motion out — one `ROBOT.md` per body, no retraining.**

[![ci](https://github.com/Hcoder10/animacy/actions/workflows/ci.yml/badge.svg)](https://github.com/Hcoder10/animacy/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![demo](https://img.shields.io/badge/demo-live%20in%20your%20browser-brightgreen)](https://hcoder10.github.io/animacy/web/)

Expressive robots ship with a menu of hand-animated moves — the Autonomous Lamp has 31, the Reachy Mini has 85 — and when the menu runs out they repeat themselves. animacy takes motion from **people** (any video, or your webcam), turns it into one canonical motion space, and maps it onto **any** robot through a single markdown file.

**▶ [Try it in your browser](https://hcoder10.github.io/animacy/web/)** — both robots on their real URDFs: vendor clips, human motion retargeted live, your webcam, and a talk mode where the robot speaks and moves in sync. No install.

<p align="center">
  <a href="https://github.com/Hcoder10/animacy/raw/main/docs/media/animacy_demo_720.mp4">
    <img src="docs/media/viewer_talk_mode.png" alt="The Autonomous Lamp and the Reachy Mini moving to the same speech" width="820">
  </a>
  <br>
  <em>▶ Demo film (3½ min) — the two robots explain the pipeline, moving with motion generated from their own voices</em>
</p>

---

## How it works

```
video / webcam ──▶ canonical human motion ──▶ ROBOT.md ──▶ your robot
                   28 channels @ 30 Hz         one file
speech ───────────────────┘  (motion generated from the same waveform, so it is in sync)
```

| step | what happens | where |
|---|---|---|
| **capture** | face + body on video become 28 channels: head 6-DoF, gaze, brows, mouth, torso, an arm, a speaking flag | [`CANONICAL.md`](docs/CANONICAL.md) |
| **declare** | your robot is one markdown file: joints, limits, rest pose, safety ceiling, a mapping | [`ROBOT_MD_SPEC.md`](docs/ROBOT_MD_SPEC.md) |
| **retarget** | gains fitted from the vendor's own clips, FK gaze compensation, springs, a hard speed cap (time stretches, never clips) | [`RETARGET.md`](docs/RETARGET.md) |
| **interact** | speech → motion at 30 Hz, on a real robot or in the page | [`MODEL.md`](docs/MODEL.md) |

## Try it in 60 seconds

```bash
git clone https://github.com/Hcoder10/animacy && cd animacy
pip install -e .                       # Python >= 3.10; add ".[capture,urdf]" for the rest

animacy check robots/lamp              # validate a robot: ROBOT.md + URDF + limits
animacy preview robots/lamp            # PNGs of the calibration poses + a sign probe

# a person talking on video -> a canonical clip, then onto two different robots
animacy capture  --source talk.mp4 -o data/clips/talk --duration 60
animacy retarget --robot lamp        --clip data/clips/talk -o out/talk.csv  --format autonomous_os_csv
animacy retarget --robot reachy_mini --clip data/clips/talk -o out/talk.json --format pollen_move

python -m http.server 8000             # the viewer at localhost:8000/web/
```

With a robot on the network: `animacy say "Hi, I'm animacy." --robot reachy_mini --url http://<robot>:8000`
and `animacy mirror --source 0 --robot reachy_mini --url http://<robot>:8000` drives it from your webcam live.

## Robots

| robot | joints | URDF | vendor clips | signs verified | on hardware |
|---|---|---|---|---|---|
| **Autonomous Lamp** | 5 | built from the vendor CAD | 31, verbatim | in sim, vs the vendor's device notes | not yet (no unit) |
| **Reachy Mini** | 9 | Pollen meshes, serial viz chain | 16 of Pollen's 85 | in sim | **yes** — [evidence](docs/evidence/reachy_sim2real_20260826.md) |
| **SO-101 arm** | 6 | vendor URDF | — | FK only | no |

**Adding a robot is one folder**: a `ROBOT.md` and a URDF. A coding agent added the SO-101 in **26 minutes** by following [`ADD_A_ROBOT.md`](docs/ADD_A_ROBOT.md), logging every unclear step — those became rules 7–10 of the spec.

## For Autonomous OS

The Lamp and the Reachy Mini are both Autonomous OS bodies, and animacy's `ROBOT.md` mirrors theirs (same joint names, same safety source). Their `docs/not-built-yet.md` asks for four things:

- **a move that plays on either body** → a move here *is* human motion; each body plays it through its own file (`--format pollen_move` / `--format autonomous_os_csv`)
- **community moves for Reachy** → capture in the browser, export Pollen's recorded-move JSON
- **recorded animations under the safety gate** → time is stretched, never clipped; 0 speed-cap violations, ever
- **a live policy behind `/policy/run`** → `animacy say --sink autonomous_os_hal` is that executor shape

Walkthrough for a Lamp owner: [`docs/AUTONOMOUS_OS.md`](docs/AUTONOMOUS_OS.md).

## Results — including what fails

We do not grade ourselves. A separate model watches short clips **blind** (no idea what produced them), next to the vendors' own hand-made animations, on **sealed** test lines. We publish what it says.

| | greeting | agreement | doubt | excitement | thinking |
|---|---|---|---|---|---|
| animacy on the Lamp | 6 | 5 | 6 | **7** | 6 |
| the Lamp's own vendor clips | 6 | 6 | 7 | 6 | 8 |

Level with hand-made animation on some movements, below on others. Our own bar was 8/10 everywhere — **not met**. Full tables, the judge's verbatim critiques and every run: [`docs/RESULTS.md`](docs/RESULTS.md), [`docs/evidence/grading/`](docs/evidence/grading/).

Also measured: the browser retargeter equals the Python one to **1e-6**; **0** speed-cap violations on every clip; a physical Reachy Mini tracks commanded head angles within a couple of degrees; **213 tests**.

The learned speech→motion model beats its floors on likelihood and motion statistics but **not** on beat alignment, so **retrieval ships as the default** and the model stays selectable. That is in [`docs/RESULTS.md`](docs/RESULTS.md) too.

## Data

73 license-verified clips, 321 valid minutes, 37 speakers, ~12 languages — public domain and CC-BY only, licence evidence stored beside every clip, and a harvester running toward 5,000 hours.

- human motion: [`squaredcuber/animacy-human-motion`](https://huggingface.co/datasets/squaredcuber/animacy-human-motion)
- for imitation learning: [`animacy-lamp-lerobot`](https://huggingface.co/datasets/squaredcuber/animacy-lamp-lerobot) · [`animacy-reachy-mini-lerobot`](https://huggingface.co/datasets/squaredcuber/animacy-reachy-mini-lerobot) (LeRobot v3.0, validated with the real loader)

## Status

**Works:** the canonical schema and the `ROBOT.md` contract; capture from video, webcam or the browser; retargeting with an exact speed guarantee; three robots; the browser demo; talk and listen modes; a physical Reachy Mini; two published datasets.

**Not yet:** no Lamp hardware — that path is verified against Autonomous's own route code and CAD, not a unit. The blind gate is failing our 8/10 bar. The learned model does not beat shuffled audio on beat alignment. SO-101 is sim only.

## Layout

```
animacy/        the library: schema, profile, retarget, capture, model, serve, sinks
robots/<name>/  ROBOT.md + urdf/ + meshes/ + clips/      docs/  the contracts and the evidence
web/            the viewer (static, no build)            scripts/  data, harvest, video, evaluation
```

## Licence

Apache-2.0. Autonomous Lamp CAD, recordings and joint conventions © Autonomous (Apache-2.0); Reachy Mini description and emotion library © Pollen Robotics / Hugging Face (Apache-2.0); SO-101 © TheRobotStudio (Apache-2.0). LeLamp (GPL-3.0) was read as a kinematic reference only — no file from it is in this repository. Video sources are public domain or CC-BY, credited per clip.

Lineage: [reachy-duplex](https://github.com/Hcoder10/reachy-duplex). Built for the Autonomous Open Source Grant, August 2026.
