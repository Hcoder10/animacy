# Documentation index

Start at the root [`README.md`](../README.md); this page is the map of everything
under `docs/`. The two **contracts** are normative — change them and you change
what every robot and every model in the project means. Everything else describes
how a part works, or records what actually happened when it was run.

## Contracts (normative)

| doc | what it fixes |
|---|---|
| [`CANONICAL.md`](CANONICAL.md) | The canonical human motion space `animacy.human.v1`: the 28 channels, their units, their sign conventions, and the clip-on-disk layout. Capture writes it, models predict it, robots consume it. Nothing else in the project may define a channel. |
| [`ROBOT_MD_SPEC.md`](ROBOT_MD_SPEC.md) | The `ROBOT.md` format every robot folder must satisfy: joint table, limits, rest pose, `max_speed`, and the linear canonical→joint mapping. `animacy check` is the executable form of this page. |

## Guides

| doc | read it when |
|---|---|
| [`ADD_A_ROBOT.md`](ADD_A_ROBOT.md) | You are adding a body. Written for a person *or* a coding agent; the goal is that no Python is needed. Worked example: [`../robots/so101/ADDING_LOG.md`](../robots/so101/ADDING_LOG.md). |
| [`RETARGET.md`](RETARGET.md) | You need the maths and the fitted numbers behind the mappings — how canonical channels become joint values, how the speed cap is enforced by time-stretching, and how the two shipped mappings were fitted to the vendors' own clip envelopes. |
| [`MODEL.md`](MODEL.md) | You want the speech→motion stack: the VQ tokenizer, the autoregressive audio→codes decoder, the retrieval baseline, and how each is exported to the browser. |
| [`LEROBOT.md`](LEROBOT.md) | You want the robot-space datasets for imitation learning — feature layout, the export command, and the Hub repos. |
| [`HARVEST.md`](HARVEST.md) | You want to grow the corpus. The licence-verified crawl/fetch/index pipeline and how to run it. |
| [`AUTONOMOUS_OS.md`](AUTONOMOUS_OS.md) | You own an Autonomous Lamp and want animacy on it: the HAL routes used, the safety ceilings, and a `PolicyService` adapter sketch. |
| [`GRADING.md`](GRADING.md) | You want the acceptance gate: how the blind judge is run, what it can and cannot see, how clips are sealed, and the pass rule. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | You are opening a PR. |
| [`../web/README.md`](../web/README.md) | You are working on the viewer. |

## Evidence and results

These are records of runs, not plans. They are the only place numbers come from.

| doc | what it records |
|---|---|
| [`RESULTS.md`](RESULTS.md) | Every measured number in the project, in chronological order, each citing a `checkpoints/<run>/REPORT.md`. Includes the results that did **not** go our way. |
| [`evidence/reachy_sim2real_20260826.md`](evidence/reachy_sim2real_20260826.md) | Reachy Mini sim-to-real on a physical unit: commanded vs read-back per axis, plus the owner's visual confirmation. Raw log alongside it as JSON. |
| [`evidence/grading/20260826_2320.md`](evidence/grading/20260826_2320.md) | Blind grading run 1 (baseline, pre-fit mappings). Gate: FAIL. |
| [`evidence/grading/20260827_0301_run2.md`](evidence/grading/20260827_0301_run2.md) | Blind grading run 2 (mapping v2, v2a bundle, intent v3). Gate: FAIL. |
| [`evidence/grading/20260827_1501_run3.md`](evidence/grading/20260827_1501_run3.md) | Blind grading run 3, the latest (gesture prototypes + energy floor). Gate: FAIL — and the vendors' own hand-authored clips score 6.6 / 5.6 on the same rubric. |

## Project

| doc | what it is |
|---|---|
| [`SUBMISSION.md`](SUBMISSION.md) | The grant-submission kit: the post copy, the demo-video shot list, and a claim→backing-file table for every sentence published. |
| [`video/script.md`](video/script.md) | The demo video script. |

## Reading order

- **Evaluating the project in five minutes:** root [`README.md`](../README.md) → [`RESULTS.md`](RESULTS.md) → the latest [grading run](evidence/grading/20260827_1501_run3.md).
- **Adding a robot:** [`CANONICAL.md`](CANONICAL.md) → [`ROBOT_MD_SPEC.md`](ROBOT_MD_SPEC.md) → [`ADD_A_ROBOT.md`](ADD_A_ROBOT.md).
- **Working on the motion model:** [`CANONICAL.md`](CANONICAL.md) → [`MODEL.md`](MODEL.md) → [`RESULTS.md`](RESULTS.md) → [`GRADING.md`](GRADING.md).
- **Putting animacy on a Lamp:** [`AUTONOMOUS_OS.md`](AUTONOMOUS_OS.md) → [`../robots/lamp/urdf/README.md`](../robots/lamp/urdf/README.md) → [`RETARGET.md`](RETARGET.md).
