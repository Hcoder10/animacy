# Blind motion grading (the acceptance gate)

The project's definition of done is not a held-out metric; it is a judge. Kimi
K3, told nothing about how a clip was made, must rate the learned model's
motion **>= 8/10 on five distinct movements, on each robot**. This document is
how that judgement is produced, what makes it blind, what the pass rule is,
and what it cannot see.

The gate is owned by `animacy/grade/`. The rubric (`rubric.py`) and the pass
rule (`run.gate`) are not to be edited by the streams they judge (model,
retarget); `tests/test_grade.py` pins both.

## Run it

```
# the venv must have playwright (+ chromium), soundfile, Pillow; ffmpeg on PATH; the kimi CLI installed
python scripts/grade_run.py --robots lamp reachy_mini --sources model retrieval envelope --seeds 2 \
    --out data/grading/<timestamp>
```

Outputs:

| path | what |
|---|---|
| `data/grading/<run>/probe.json`, `probe_reel.mp4` | the video-ability probe and its answer |
| `data/grading/<run>/tts/*.wav` | the utterance audio (Windows SAPI via `animacy.tts`, 16 kHz) |
| `data/grading/<run>/clips/*.json` | every joint table that was rendered, by origin id |
| `data/grading/<run>/reels/*.mp4` (+`parts/`) | the reels the judge watched, per-clip parts |
| `data/grading/<run>/manifest_sealed.json` | blind number -> origin. **Never copied to the judge.** |
| `data/grading/<run>/kimi/*.json` | raw judge responses per reel (cache: a re-run reuses valid ones) |
| `data/grading/<run>/results.json` | unsealed records, summary, gate, calibration, consistency |
| `docs/evidence/grading/<run>.md` | the committed summary (no videos) |

`data/grading/` is gitignored (videos, wavs); only the markdown summary is
committed. Exit code is 0 only when every robot passes.

Useful flags: `--no-kimi` (build clips, reels and the sealed manifest without
calling the judge), `--software` (SwiftShader render, ~8 fps instead of ~100
fps on the GPU), `--no-speech-strip`, `--max-reel-seconds`, `--parallel`,
`--seed` (blind order), `--force-probe`, `--gate-source auto|model|retrieval|envelope`,
`--seeds-deterministic M`, `--label "..."`, `--compare <baseline run_dir>`,
`--slow-variant` (adds the 0.5x sub-run after the gated run), `--speed 0.5`
(render one whole run slow), `--report-only <run_dir>` (regenerate the
markdown summary from an existing `results.json`; `--label`/`--compare`
apply). Judge responses are cached per reel, so a re-run into the same
`--out` re-renders the clips but only repeats the judge calls that failed.
`scripts/grade_queue.py` can arm a run to fire automatically when a mapping
commit and a new bundle land (`--dry-run` prints the condition status).

Robustness: a run refuses to start if another live process holds
`<out>/RUNNING.pid` (two renderers in one run dir once raced and both
died), and transient stalls are retried with backoff and logged as
`[retry]`: the viewer page load (4 attempts, fresh page each time), frame
capture (3), card render (4, 60 s screenshot timeout) and judge calls (3,
30/120/300 s). A network outage during run 3 killed a run at the card
render before this existed.

A single clip can be rendered with `python web/dev/render_clip.py --robot lamp
--native nod --out out/nod.mp4` (also `--csv`, `--table`, `--audio`).
`python scripts/grade_showcase.py --run data/grading/<run> --robot lamp
--source retrieval --out data/grading/showcase_lamp` re-renders a run's
best-scoring clip per movement cleanly (no card, no strip, with the utterance
audio) and writes `showcase.json` with the scores; winners from the sealed
held-out set land under `SEALED_heldout/` with a README, because their audio
is the sealed line and publishing one burns the gate.

## What is graded

Five movements, each an utterance chosen to elicit a distinct behaviour
(`movements.py`):

| key | the robot says | vendor calibration clip (lamp / reachy) |
|---|---|---|
| greeting | "Hey! Good to see you again." | greeting / welcoming1 |
| agreement | "Yes, exactly, that is what I meant." | nod / yes1 |
| doubt | "Hmm, no, I really don't think that's right." | headshake / no1 |
| excitement | "No way, that is incredible news!" | excited / amazed1 |
| thinking | "Let me think about that for a second... okay." | confused / thoughtful1 (the lamp's `thinking_deep.csv` is a constant pose: every joint range is 0.0 over its 15 s, so it cannot calibrate anything) |

For every movement, on every robot, a candidate clip is produced from each
source in `animacy.serve.SOURCES` (`model`, `retrieval`, `envelope`) with 2
seeds, exactly as the talk loop does it: TTS waveform -> source ->
`retarget_clip` through the robot's `ROBOT.md`. Retrieval is deterministic,
so its two seeds are the same clip; the gap between their scores is reported
as the judge's own noise. The vendors' hand-authored clips for the same
intents (long clips trimmed to their most active 6 s window) are rendered through the same pipeline as
calibration: if the judge does not rate them well, the rendering or the
rubric is broken, not the candidates.

## Rendering

`render.py` drives the web viewer headlessly (Playwright + Chromium, GPU via
ANGLE/D3D11, SwiftShader fallback): the repo is served by a local
`http.server`, `web/` is opened, all UI is hidden by injected CSS, and for
each frame the URDF joint values (`animacy.retarget.to_urdf_values`) are set on
`window.animacy.robots[<robot>].viewer` directly, rendered, and read back with
`canvas.toDataURL`. Frame *i* of the video is row *i* of the joint table on a
30 fps grid; the browser's clock never touches the motion. ffmpeg encodes
H.264 at 512x512 with the utterance audio muxed in (AAC, 1 s offset for the
card). Fixed 3/4 front camera per robot (the viewer's `iso` view, camera
distance x0.9).

Each clip = 1 s title card ("Clip N" and one line: `The robot says: "..."` or
`The robot expresses: <intent>`) + the clip + 0.5 s black. A thin loudness
strip of the utterance with a moving marker is burned along the bottom edge,
because the judge cannot hear the audio track (see below); silent clips get a
flat strip. Nothing else is drawn.

## Blindness

1. **Numbering.** Per robot, all clips (candidates and vendor alike) are
   shuffled with a seeded RNG and numbered 1..N in that order; reels are
   consecutive runs of numbers (`reel.py`).
2. **The sealed manifest** (`manifest_sealed.json`, number -> source, seed,
   movement, vendor clip) is written next to the run outputs. The judge's
   workspace is a fresh directory under `<OS temp>/motion_judge/<run>/`
   containing only the reel (and, transiently, the prompt file);
   `run.check_workspace` refuses to proceed if anything else is there, and the
   listing is recorded before every call (`results.json` ->
   `judgements[*].workspace_listing_before`). The reel path is quoted in the
   prompt, so the directory name is checked against `FORBIDDEN` too (the
   first attempt of the first run was stopped by exactly this check: the
   workspace used to live under `animacy_grade/`).
3. **The rubric** (`rubric.py`) carries no project context. `FORBIDDEN` lists
   the words that may never appear (animacy, model, retrieval, generated,
   vendor, learned, training, envelope, heuristic, baseline, dataset,
   canonical, retarget, seed, ...); `build_prompt` is tested against it for
   every robot, and the run aborts if a prompt hits the list.
4. **Nothing in the video says how a clip was made.** Card lines only carry
   the sentence or the intent. The one asymmetry is inherent: vendor clips are
   silent and use "expresses" instead of "says", so a judge could tell them
   from candidates as a population; that is why they are reported as
   calibration and never compared head-to-head with candidates.
5. **Independence.** The judge is asked to score each clip on its own, to
   describe it before scoring (the description proves it watched), and not
   to compare or rank.

The rubric's dimensions (1-10): lifelike, intent, timing, physical, appeal,
plus overall and a one-line reason; JSON only; retried up to 3 times on a
malformed response (`rubric.validate_response`).

## The pass rule

For each robot separately: the **source under test** must have **overall >=
8.0 on all five movements, using the mean over seeds**. Best-of-seeds is
reported ("would pass") for information and never decides. A movement with no
scored clip fails. Vendor calibration mean < 6 marks the run as broken. The
rule is `animacy.grade.run.gate`; `tests/test_grade.py` checks it on fixture
scores (passes at 8.0 exactly, fails at 7.9, fails when best-of-seeds would
pass but the mean does not, ignores other sources' scores, per-robot).

The source under test is the **shipped default**: `--gate-source auto`
(default) reads `default_backend` from `web/models/model.json`, i.e. what a
user of the web demo gets (run 1 was made when the definition was the
`model` source and records that; from run 2 on it is the bundle's
`default_backend`, `retrieval` at the time of writing). The report names the
source under test, and applies the same rule to every other graded source
for information. `--gate-source model` pins it explicitly.

**Lines under test (from run 2): the sealed held-out set.** The five
utterances in `movements.py` are known to every agent (they even appeared in
the intent lexicon), so from run 2 the gate is scored on five **held-out**
lines with the same intents, authored by the grader and stored only in
`data/grading/heldout_lines.json` (gitignored; never messaged to another
agent; `load_heldout_movements` refuses a file that shares a 3-word phrase
with a tuning line). Both sets are rendered and judged: the held-out set
decides (`--gate-lines auto` picks it when the file exists), the original
five are reported alongside as the **tuning set**, and the report prints
`tuning - heldout` per robot for the source under test: a large positive gap
means the motion was tuned to the known lines, not to the intents. Held-out
clips carry ids like `lamp/greeting@heldout/model/s0`; the judge sees their
text on the card, but the report redacts the sealed lines (exact matches and
any shared 3-word run) from the judge's descriptions and notes. Vendor
calibration clips are rendered once and shared by both sets.

`--require-lexicon intent.v2` refuses to start until the shipped bundle's
`intent.lexicon_version` says the tuning lines were stripped from the lexicon.

Seeds: `--seeds N` for the stochastic source (`model`), `--seeds-deterministic
M` (default 1) for `retrieval` and `envelope`, whose seed does not change the
clip (retrieval ignores it; envelope's only shifts slow drift phases). Run 1
used 2 seeds everywhere and measured the judge's noise on the identical
retrieval pairs (mean |gap| 0.4 on the lamp, 0.8 on the Reachy).

Intent: each movement carries an intent tag (`Movement.intent_tag`, e.g.
`greeting`). For the `model` and `retrieval` sources the grader hands over
exactly what `animacy say "<line>" --intent <tag>` hands over:
`animacy.model.intent.analyse(line, override=tag)` (the tag's base arousal
plus the line's punctuation), and only when the source's signature accepts
`intent` (`movements.accepts_intent`); the envelope heuristic and older
signatures are untouched. The clip record's `meta.intent_passed` says
whether it was used and `meta.intent` records the arousal/amplitude that
resulted. Note for readers of run-2 numbers: the intent lexicon in
`animacy/model/intent.py` was written with the grader's five lines in it
verbatim, so the tag is guaranteed to resolve for these utterances; whether
it generalises to other lines is a separate question this gate does not
answer.

## What the judge can and cannot see (measured)

Probe (`probe.py`, re-run every grading run, cached in `probe.json`): a 3 s
reel with the card "Clip 7 / The robot expresses: a nod" and the lamp's own
`nod` clip. Kimi K3 via the local CLI (`ReadMediaFile`, native `video_in`):

- **Video: yes.** It read the card verbatim and described "tilts its head up
  and down in a nodding motion", direction "up-down".
- **Sampling: ~2 frames per second natively, 6 fps when it digs in.** It
  reported 6 sampled frames "about 0.5 s apart" for the 3 s reel. On a 67 s
  ten-clip calibration reel it reported ~120 native samples for the whole
  file and then, on its own initiative, re-extracted every clip as 6 fps
  frame grids with its shell (~450 frames in total) and said "fine easing
  detail (small anticipation/settle) is partially limited by the 6 fps
  sampling". Micro-motion, overshoot/settle and jitter are therefore only
  partly observable; `lifelike` and `physical` are judged from sampled poses.
  Reels are kept under ~90 s.
- **Cost: ~5-6 minutes per 10-clip reel** (the calibration reel took 338 s);
  6-15 minutes per reel under machine load in run 1 (348-889 s).
- **What it actually did in run 1** (its own notes, verbatim in the report):
  for every reel it extracted frames itself at 4-10 fps into contact sheets,
  spot-checked full-resolution frames, ran a 30 fps per-frame pixel-difference
  motion profile and head-blob tracking (jitter residual, teleport check),
  and read the loudness strip per clip to align motion beats with speech. So
  `physical` is judged on real 30 fps evidence; easing detail is still
  judged from 4-10 fps samples.
- **Audio: no.** It confirmed the AAC track exists (with ffprobe) but cannot
  listen to it. **The judge grades from video only.** `timing` therefore
  measures rhythm plausibility against the transcript card and the burned-in
  loudness strip, not audio sync. It stays in the rubric because a motion
  that ignores the voice's phrasing is still visible in the strip; the pass
  rule uses `overall` only, and the report prints per-dimension means by
  source so it is visible whether `timing` is what drags a score.

## Known limitations

- The judge is a sampled-frame viewer, not a 30 fps one; scores on the
  fine-motion dimensions are upper-bounded by what 2 fps shows. A judge with
  denser sampling would be a stronger gate, not a different one.
- The judge cannot hear speech; sync is judged visually via the strip.
- At ~2 samples per second a 0.5 s event (a brow flick, an antenna snap, a
  quick nod) can fall between the judge's samples entirely. `--speed 0.5`
  renders a slow-motion variant (the card says "slow motion (0.5x)" and
  nothing about why; audio is time-stretched so the strip stays aligned) to
  ask that question in a SEPARATE run; its numbers are never mixed into, or
  compared with, a speed-1 gate run.
- Kimi's scores are noisy; the retrieval seed-pair gap is the measured noise
  floor per run. The gate uses means over seeds, and more seeds tighten it.
- Rendering fidelity: URDF meshes, no motion blur; the lamp's joint values are
  the vendor's servo units (within ~10% of degrees, `robots/lamp/ROBOT.md`).
- The judge has a shell; it could in principle look outside its workspace,
  but the manifest lives elsewhere, is never named, and the prompt gives it
  no reason to look. Workspace listings are recorded for audit.
- Windows SAPI is the TTS in this run; the browser demo uses Kokoro. The
  motion is driven by the same 16 kHz features either way.

## Results

Every run writes `docs/evidence/grading/<run>.md` with: what the judge could
see, the gate (source under test, plus the same rule on every source for
information), the vendor calibration line ("the vendor's own clips score X on
this rubric"), overall by robot x source x movement with per-seed values,
per-dimension means by source, judge self-consistency, provenance (git HEAD,
hashes of `ROBOT.md`/`retarget.py`/`model.json`/checkpoint files, copies under
`<run>/provenance/`), every clip unsealed with the judge's description, the
judge's per-reel notes verbatim, and, with `--compare <baseline run>`, a
side-by-side table baseline -> this run (delta) per robot x source x movement.

`--slow-variant` renders and judges the same clips again at 0.5x as a
separate sub-run (`<out>_slow`, cards say "slow motion (0.5x)"), compared
against the normal-speed run; it answers whether under-sampled easing
depresses scores and is never a gate.

`--variant NAME=SOURCE:KEY=VALUE` (repeatable) adds an A/B column: the base
source called with a knob, e.g. `retrieval_p0=retrieval:proto_weight=0`. The
knob is applied only when it is an explicit parameter of the source function
(a `**kw` catch-all does not count, because a swallowed knob would make the
A/B a silent no-op); otherwise the run says so in its notes and the column
equals the base. After the clips are built, every variant clip is compared
numerically with its base clip (same robot, movement, line set, seed): a
variant whose clips are all identical is dropped before rendering and the
report says so, so no judge call is spent on a duplicate column; partial
identity is counted per clip. A variant can never be the gate source.

Every report also carries **"What the judge keeps saying (verbatim, by
dimension)"**: for lifelike, timing and appeal, how many candidate clips
scored <= 5 (by source), the words that recur in the judge's reasons for
them, and its reasons quoted verbatim, lowest scores first and spread over
robots and sources. That section is what the model and mapping streams work
from (in run 2 it read, for the lamp's learned source: "essentially a still
image behind the dialogue", "a frozen hold under an affirmative line").

**Run 1 (baseline)**: `docs/evidence/grading/20260826_2320.md`, made before
the fitted mappings (`d73ced7`/`37eca54`) and retarget v1.1, on
`checkpoints/v1` (arch ff), gate on `model`: lamp FAIL (model 6.5 / 5.5 /
4.5 / 4.0 / 5.0), Reachy FAIL (5.5 / 5.5 / 6.0 / 5.5 / 6.0); the vendor's
own clips score 6.6 on both robots; judge noise 0.4-0.8 points.

**Run 2**: `docs/evidence/grading/20260827_0301_run2.md`, mapping v2
(whole-arm, `eae7853` + settle `6976777`), `checkpoints/v2a` (AR), intent
v3, gate = shipped default `retrieval` on the sealed held-out lines: lamp
FAIL (6.0 / 5.0 / 6.0 / 7.0 / 5.0), Reachy FAIL (6.0 / 7.0 / 7.0 / 6.0 /
7.0, retrieval now at vendor level there); tuning-minus-heldout gap +0.2 /
0.0 (no contamination); vendor calibration 6.2 / 6.0; held-out tag
resolution 4/5. The learned source on the lamp went from "too restless"
(run 1) to "essentially a still image" (run 2).
