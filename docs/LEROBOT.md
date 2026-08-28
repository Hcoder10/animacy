# LeRobot export (`animacy.lerobot_export`)

Turns captured clips into a **LeRobot v3.0 dataset for one robot**, so the
motion becomes VLA / imitation-learning training data in the format the
`lerobot` stack reads natively.

```
python scripts/export_lerobot.py --robot lamp        --clips data/clips --out data/lerobot/animacy_lamp        --validate --push squaredcuber/animacy-lamp-lerobot
python scripts/export_lerobot.py --robot reachy_mini --clips data/clips --out data/lerobot/animacy_reachy_mini --validate --push squaredcuber/animacy-reachy-mini-lerobot
```

Published datasets (public), **as pushed from the 6-clip v1 corpus**:

- https://huggingface.co/datasets/squaredcuber/animacy-lamp-lerobot — 53 episodes, 26 774 frames (14.9 min), 5 joints
- https://huggingface.co/datasets/squaredcuber/animacy-reachy-mini-lerobot — 53 episodes, 26 812 frames (14.9 min), 9 joints

> **These two datasets are behind the corpus.** They were exported from the six
> clips the v1 model trained on; the human corpus has since grown to 73 clips /
> 320.8 valid minutes / 37 speakers (`docs/RESULTS.md`, and `docs/HARVEST.md`
> for the ongoing crawl). Re-running the two commands above against the current
> `data/clips` re-exports and re-pushes both. Every episode count, frame count
> and task-string count in this document describes the pushed v1 export unless
> it says otherwise.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("squaredcuber/animacy-lamp-lerobot")      # downloads the v3.0 tag
item = ds[0]
item["observation.state"]          # (5,)  float32 lamp joints, degrees
item["action"]                     # (5,)  next-frame joints
item["observation.audio_features"] # (66,) log-mel + energy, the animacy.features contract
item["observation.speaking"]       # ()    0/1
item["observation.human"]          # (14,) canonical human channels (docs/MODEL.md)
item["task"]                       # "speaking: president obama's weekly address"

# action chunks the way ACT / Diffusion consume them
ds = LeRobotDataset("squaredcuber/animacy-lamp-lerobot", delta_timestamps={"action": [i / 30 for i in range(30)]})
ds[0]["action"].shape              # (30, 5)
```

## Format version

| | |
|---|---|
| `codebase_version` | **v3.0** (`meta/info.json`) |
| written to match | `lerobot` **0.6.1** (`lerobot.datasets.dataset_metadata.CODEBASE_VERSION == "v3.0"`) |
| validated by | the real `LeRobotDataset` in a separate venv (`.venv-lerobot`, `lerobot[dataset]==0.6.1`, torch 2.11 cpu, datasets 4.8.5, pyarrow 25) |
| videos | none — `video_path: null`; the dataset is state/audio only (lerobot does not require videos) |

The writer is pyarrow/pandas only: `animacy` never imports `lerobot`. The
on-disk tree is exactly what lerobot's own `DatasetWriter` produces for a
video-less dataset:

```
<out>/
  meta/info.json                            fps, robot_type, features, totals, chunking, splits
  meta/stats.json                           per feature: min/max/mean/std/count + q01,q10,q50,q90,q99
  meta/tasks.parquet                        pandas frame, index "task" -> task_index
  meta/episodes/chunk-000/file-000.parquet  one row per episode: episode_index, tasks, length,
                                            data/chunk_index, data/file_index, dataset_from_index,
                                            dataset_to_index, meta/episodes/*, stats/<feature>/<stat>,
                                            plus animacy/* provenance columns (clip, run frames,
                                            src_frame_start/end, stretch, speaking_fraction, license, url)
  meta/animacy.json                         export provenance (lerobot ignores it)
  data/chunk-000/file-000.parquet           one row per frame, one parquet row group per episode,
                                            vectors as fixed_size_list<float32>[d]  (== datasets.Sequence(length=d))
  README.md                                 dataset card (written on --push)
```

Files roll over to `file-001` after `data_files_size_in_mb` (100 MB, lerobot
default); at ~1 kB/frame the current corpora are one file each.

## Features

`observation.state` / `action` are in the robot's own units and joint order
from its `ROBOT.md` (lamp: degrees; Reachy Mini: mm for `head_x/y/z`, degrees
otherwise). Everything below is per frame at `fps` (30).

| key | dtype | shape | names | meaning |
|---|---|---|---|---|
| `observation.state` | float32 | (J,) | `ROBOT.md` joint names | robot joints from `retarget_clip` (speed-legal, smoothed) |
| `action` | float32 | (J,) | same | `observation.state[t+1]` inside the run; the run's last frame holds |
| `observation.human` | float32 | (14,) | `animacy.model.data.MODEL_CHANNELS` | canonical human channels the robot frame was computed from (NaN -> 0) |
| `observation.audio_features` | float32 | (66,) | `mel_00..mel_63, log_energy, delta_log_energy` | `animacy.features.audio_features` on the 30 Hz grid, per-clip normalised; zeros when the clip has no audio |
| `observation.speaking` | float32 | (1,) | – | VAD flag, 0/1 |
| `observation.environment_state` | float32 | (67,) | audio names + `speaking` | duplicate of `[audio_features, speaking]` under the key lerobot's ACT / Diffusion require when there is no camera (`--env-state audio|human|none`) |
| `timestamp` | float32 | (1,) | – | `frame_index / fps` |
| `frame_index`, `episode_index`, `index`, `task_index` | int64 | (1,) | – | lerobot bookkeeping |

`J` = 5 (`base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch`) for the
lamp, 9 (`head_x, head_y, head_z, head_roll, head_pitch, head_yaw, body_yaw,
antenna_left, antenna_right`) for Reachy Mini. `lerobot`'s
`dataset_to_policy_features` types them as: `observation.state` STATE,
`action` ACTION, `observation.environment_state` ENV, the other
`observation.*` STATE (ignored by policies that only read `observation.state`).

## Episodes

- One episode per contiguous `face_valid` run of **>= 3 s** (`--min-seconds`);
  runs longer than **20 s** (`--max-seconds`) are split into equal pieces.
  Frames where the face was not tracked never enter the dataset.
- The robot trajectory is `animacy.retarget.retarget_clip` on that run: mapping
  arithmetic from `ROBOT.md`, time *stretched* where a joint would exceed
  `max_speed`, resampled to the robot grid, zero-phase smoothed, clamped to
  joint limits. Because time can stretch, the human-side columns are re-aligned
  onto the robot grid by picking, for every robot frame, the source frame
  nearest in stretched time (`animacy/src_frame_start/end` per episode). On the
  six v1 clips the stretch is <= 1.01x everywhere except one 4 s run of
  the low-quality `sd_rapper_interview`, which `--max-stretch 1.1` (default)
  drops — audio that no longer plays at real time would mislead a speech-driven
  policy.
- Task text is `"<role>: <what>"`: role from `meta.role`, else the VAD majority
  of the episode (`speaking` / `listening`); `what` from `meta.prompt` /
  `meta.task`, else the cleaned `meta.title` (extension and date prefix
  stripped, lower-cased, <= 64 chars). Both Obama clips therefore share
  `speaking: president obama's weekly address`; the pushed v1 export has 5
  distinct task strings.

The pushed v1 export (both robots, `--max-stretch 1.1`) — six clips; see the
note at the top of this page:

| clip | license | episodes | frames (lamp / reachy) |
|---|---|---|---|
| obama_2015_02_07 | Public Domain | 9 | 5341 / 5348 |
| obama_2014_09_13 | Public Domain | 12 | 6708 / 6710 |
| kende_interview_2014 | CC-BY-3.0 | 12 | 6112 / 6112 |
| royal_society_cloke | CC-BY-3.0 | 16 | 7253 / 7290 |
| cbp_vlog_day2 | Public Domain | 2 | 783 / 785 |
| sd_rapper_interview | Public Domain | 2 | 577 / 567 |

## Stats

`meta/stats.json` holds, per feature, `min`, `max`, `mean`, `std` (population),
`count` (`[total_frames]`) and exact quantiles `q01, q10, q50, q90, q99` — the
keys `lerobot.datasets.compute_stats` produces, so `lerobot`'s normalisation
(`MEAN_STD`, `MIN_MAX`, `QUANTILES`) works unchanged. The same stats are also
stored per episode in the episodes parquet (`stats/<feature>/<stat>`), which
`lerobot`'s aggregation/edit tools read.

## Validation

`--validate` (implied by `--push`) runs the real loader in the lerobot venv and
fails the export if anything is off:

```
ANIMACY_LEROBOT_PYTHON=<venv python>   # optional; default .venv-lerobot/{Scripts/python.exe,bin/python}
python scripts/export_lerobot.py --robot lamp --out data/lerobot/animacy_lamp --force --validate
```

What it checks (`animacy.lerobot_export.VALIDATOR_SRC`): `LeRobotDataset(repo_id,
root=out)` loads; `codebase_version == CODEBASE_VERSION`; parquet column types
equal `get_hf_features_from_features(features).arrow_schema`; four items have
the declared shapes/dtypes and are finite; `timestamp == frame_index / fps`;
`action[t] == state[t+1]`; stats shapes; `delta_timestamps` action chunking
with `_is_pad`; a shuffled `DataLoader` batch of 8; `dataset_to_policy_features`
typing. The validator runs python with `-I` because this machine exports a
`PYTHONPATH` into another venv, which otherwise shadows the lerobot venv's
packages. `tests/test_lerobot_export.py` runs the writer contract on synthetic
clips in the main venv and the lerobot load when the venv exists.

Setting up the validation venv (never install `lerobot` into the main venv —
it pins torch):

```
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/Scripts/python.exe "lerobot[dataset]==0.6.1"   # + "lerobot[training]" to train
```

## Training a policy on it

lerobot 0.6.1's ACT and Diffusion policies refuse to train without an image or
`observation.environment_state`; the exporter's default `--env-state audio`
supplies the latter (speech features + speaking flag), so the policy learns
*speech -> robot motion* conditioned on the current joints. Small ACT on CPU:

```
python -I -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=squaredcuber/animacy-lamp-lerobot \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --policy.dim_model=128 --policy.dim_feedforward=512 --policy.n_heads=4 --policy.n_encoder_layers=2 \
  --output_dir=data/lerobot/train/act_lamp --job_name=act_lamp \
  --steps=200 --batch_size=16 --log_freq=10 --num_workers=0 \
  --save_checkpoint=false --wandb.enable=false --env_eval_freq=0
```

(`lerobot-train` is the same entry point; add `--dataset.root=<local out dir>`
to train from a local export instead of the Hub, `--policy.device=cuda` and
drop the size overrides for a real run. For Reachy Mini swap the repo id.)

### 200-step smoke run (lamp, CPU)

Exactly the command above (`lerobot` 0.6.1, torch 2.11 cpu, 2026-08-26): ACT
with 1.48 M parameters, batch 16, lr 1e-5, 200 steps in 37 s (5.3 step/s).
`loss` is lerobot's total (L1 + 10 x KLD, the VAE default); `l1_loss` is the
normalised action-chunk error that matters.

| step | loss | l1_loss | kld_loss |
|---|---|---|---|
| 10 | 70.98 | 0.904 | 7.008 |
| 50 | 25.48 | 0.768 | 2.471 |
| 100 | 10.47 | 0.645 | 0.982 |
| 150 | 6.53 | 0.587 | 0.595 |
| 200 | 5.46 | 0.543 | 0.491 |

Loss falls monotonically (the total is dominated by the KLD term collapsing;
L1 goes 0.90 -> 0.54 in 0.12 epochs). This only shows that the dataset feeds
lerobot's training loop end to end — it is not a trained policy, and no
held-out evaluation was run.

## What is *not* in the dataset

- No video/images: the exporter has nothing to encode, and `video_path` is
  null. A camera-conditioned policy needs a different capture.
- No `next.reward` / `next.done`: imitation only.
- Robot-space values are whatever the robot's `ROBOT.md` mapping produced at
  export time — re-export after changing a mapping or a sign.

## cli.py

`animacy lerobot` is not wired into `animacy/cli.py` yet; the subcommand would
be:

```python
p = sub.add_parser("lerobot", help="export clips as a LeRobot v3.0 dataset for one robot")
p.add_argument("--robot", required=True)
p.add_argument("--clips", default="data/clips")
p.add_argument("--out", required=True)
p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--mode", default="default")
p.add_argument("--exclude", default="")
p.add_argument("--env-state", default="audio", choices=("audio", "human", "none"))
p.add_argument("--max-stretch", type=float, default=1.1)
p.add_argument("--force", action="store_true")
p.add_argument("--validate", action="store_true")
p.add_argument("--push", default=None)
p.set_defaults(func=_cmd_lerobot)   # -> animacy.lerobot_export.export / validate_with_lerobot / push_to_hub
```
