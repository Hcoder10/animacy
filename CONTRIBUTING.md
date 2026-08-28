# Contributing to animacy

Apache-2.0. By contributing you agree your work ships under that licence.

The whole project is arranged so the two most useful contributions need no
Python at all: **adding a robot** is one Markdown file plus a URDF, and
**adding data** is running a capture command.

## Setup

```bash
git clone https://github.com/Hcoder10/animacy && cd animacy
pip install -e ".[capture,urdf,dev]"
python -m pytest -q            # 213 passed, 5 skipped on a clean checkout
```

Extras: `capture` (MediaPipe + OpenCV + audio, for `animacy capture` / `animacy
mirror`), `urdf` (yourdfpy + trimesh + matplotlib, for `animacy check`'s URDF
limit test and `animacy preview`), `robot` (requests, to talk to a real robot),
`train` (torch + ONNX). Everything optional is guarded with
`pytest.importorskip`, so a partial install skips rather than fails.

## Adding a robot — the main path

Read **[`docs/ADD_A_ROBOT.md`](docs/ADD_A_ROBOT.md)**. In short:

1. `cp -r robots/_template robots/<name>` and drop a URDF in `urdf/`.
2. Fill the joint table from the vendor's spec — names, units, limits, rest
   pose, and `max_speed` from the vendor's *safety* file.
3. Write the `default` mapping **by function, not anatomy**: gaze → whatever
   points the face, lean → base joints, affect → the body's most legible
   expressive channel.
4. `animacy check robots/<name>` until it passes.
5. `animacy preview robots/<name>` — it renders the calibration poses through
   *your* mapping to PNGs and prints what +10 units on each joint does to the
   head. Read those before opening a browser. Fix directions with `gain: -1`.
6. `animacy profile export robots/<name>`, open the viewer, play the vendor's
   own clips first — if *those* look wrong, the URDF axes are wrong, not the
   mapping.
7. Add a row to the README robots table and open a PR.

The rules that matter:

- **Never edit captured data to fix a sign.** Signs live in `ROBOT.md`.
- **Never raise `max_speed` above the vendor's safety ceiling** because the
  motor can go faster. Time is stretched, never clipped, so nothing is lost.
- **Joint names are an ABI** once merged — exports are 1:1 with vendor names.
- **If you had to write Python to add your robot, that is a bug in the spec.**
  Open an issue; the next robot should need only the `.md`.

## Adding data — the Record-mode flow

Motion data enters the project in exactly one shape: a canonical clip
directory (`motion.parquet` + `audio.wav` + `meta.json`, see
[`docs/CANONICAL.md`](docs/CANONICAL.md)). Three ways in:

```bash
# 1. a video file or your webcam
animacy capture --source talk.mp4 -o data/clips/talk --duration 60
animacy capture --source 0        -o data/clips/me   --preview

# 2. the viewer's Record mode -> a zip -> a standard clip dir
animacy import-browser take.zip -o data/clips/take

# 3. a licence-verified crawl (see docs/HARVEST.md)
python scripts/fetch_sources.py ...
```

Then it flows out unchanged through everything else: `animacy retarget` to a
robot, `animacy lerobot` to a LeRobot v3.0 dataset, `animacy.model.train` to a
checkpoint.

**Licence policy is enforced, not advisory.** Only public-domain and CC-BY
sources are fetched, the licence is verified from metadata, ND is refused, and
the evidence is copied into every clip's `meta.json`. `scripts/push_hf.py`
refuses to publish a clip with no licence record. Your own webcam clips carry
`license: self` (CLI) or CC-BY-4.0 (browser Record mode). **Do not add a clip
you cannot point at a licence for.**

## Code changes

- `animacy/schema.py:CHANNELS` is the single source of truth for the canonical
  frame. Adding or reordering a channel is a breaking change to every profile,
  every export and the browser bundle — raise an issue first.
- `animacy/retarget.py` and `web/js/retarget.js` are two implementations of one
  spec and are held equal to 1e-6 by `tests/test_web_retarget_parity.py`. Change
  one, change the other, in the same PR. The same goes for
  `animacy/features.py` ↔ `web/js/features.js`.
- **Do not weaken a test to make it pass.** In particular
  [`animacy/grade/`](animacy/grade) owns the blind acceptance gate and its pass
  rule; the rule is deliberately stricter than parity with the vendors' own
  hand-authored clips, and it currently **fails**. Report the number, do not
  move the bar. Same for the speed-cap checks in `animacy/export.py`.
- New numbers in a doc need a file to cite — a `checkpoints/<run>/REPORT.md`, an
  evidence file, or a test. [`docs/RESULTS.md`](docs/RESULTS.md) records results
  that went badly as readily as ones that went well; keep it that way.

## Tests and CI

```bash
python -m pytest -q                       # everything (heavy optional paths skip)
python -m pytest tests/test_core.py -q    # one file
```

`tests/test_lamp_hal.py` is an integration test against a running Autonomous OS
HAL and is **skipped unless one answers at `LAMP_HAL_URL`** (default
`http://127.0.0.1:5001`); `scripts/lamp_hal_sim_start.sh` boots the vendor's
laptop simulator. One assertion in it is currently failing — see the root
README's *Status* section. `tests/test_mirror_pacing.py` measures a real clock
and can be flaky on a loaded machine; re-run it alone before believing a failure.

CI (`.github/workflows/ci.yml`) has two lanes: `core` validates every
`robots/*/ROBOT.md`, runs the suite with base deps, and checks the JS↔Python
retargeter parity under Node; `optional-deps` installs CPU torch plus the URDF
extras and exercises forward kinematics, the headless `animacy preview` renders,
and the model + ONNX export.

## Pull requests

Small and self-describing. Say what you verified and how; if you could not
verify something, say that instead — an honest gap is worth more here than a
claim without a file behind it.
