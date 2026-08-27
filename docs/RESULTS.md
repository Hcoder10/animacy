# Results (updated 2026-08-27, model v1)

Every number below comes from `checkpoints/<run>/REPORT.md`, written by
`python -m animacy.model.train` with the exact command recorded in the report.
Nothing here is claimed without that file. Held-out means a **speaker the
model never saw**, evaluated on all of that speaker's valid frames.

## Data

6 license-verified clips, 20.5 min, 14.95 min face-valid with audio
(`scripts/fetch_sources.py`; sources + evidence in the HF dataset card
[squaredcuber/animacy-human-motion](https://huggingface.co/datasets/squaredcuber/animacy-human-motion)).
Two low-quality clips (multi-shot b-roll, 400×300 handheld) are excluded from
training. **This is a small dataset; a 10× larger fetch is in progress.**

## Tokenizer (VQ-VAE, 512 × 64, one code per 2 frames)

| run | codes used (train) | perplexity | val MAE (std units) | predict-mean baseline |
|---|---|---|---|---|
| v1 (hold out kende) | 512 / 512 | 460 | 0.35 | 0.71 |

Round trip on the held-out speaker: r = 0.8–0.9 on head/torso/brows. Dead-code
revival plus data-dependent initialisation were required: the stock EMA
quantiser collapsed to one code on this data size.

## Audio → motion, held out (two different held-out speakers)

| metric (held-out speaker) | kende (v1, shipped) | obama_2015 (`v1_holdout_obama2015`) |
|---|---|---|
| code NLL: model / **unigram floor** | 6.276 / **6.162** (not below) | **6.031** / 6.167 (below the floor) |
| head-beat recall: model / **shuffled audio** | 0.644 / 0.613 (margin 0.03, inside noise) | **0.617** / 0.497 (margin 0.12) |
| head-beat precision: model / shuffled | 0.384 / 0.370 | **0.52** / 0.40 |
| retrieval beat recall / shuffled | 0.526 / 0.526 | 0.576 / 0.534 |
| stillness (frames < 5°/s): model / retrieval / **truth** | 0.013 / 0.060 / **0.097** | 0.025 / – / **0.103** |
| velocity-histogram W1 (rel.): model / retrieval | 1.94 / 1.27 | 0.30 / 0.20 |
| retarget legality (speed-cap violations), lamp + reachy | 0 after the `rate_limit` fix | 0 |

Diagnostic on the obama hold-out (`scripts/model_diag.py`): the model's
expected decoded motion correlates with the unseen speaker's ground truth at
r = 0.48 on `head_pitch` (−0.02 with shuffled audio), 0.40 on `head_z`, 0.40 on
`mouth_open`, 0.22 on `head_x` — it learns speech-driven nodding and leaning.

**Verdict, stated plainly:** on one held-out speaker the learned model beats
both its shuffled-audio control and the unigram floor; on the other it does
not (that capture carries little audio-driven motion). The generated motion is
too restless on both (stillness 0.01–0.03 vs 0.10) because codes are sampled
independently per step. Therefore **retrieval ships as the default source**
(`web/models/model.json: default_backend = retrieval`), the learned model is
selectable, and v2 is an autoregressive decoder trained on the larger dataset —
its numbers will be appended here, whatever they are.

## Sim-to-real (Reachy Mini, physical unit)

`docs/evidence/reachy_sim2real_20260826.md`: canonical clip → `ROBOT.md` →
daemon at 30 Hz; every commanded axis read back within a few degrees; owner
confirmed directions visually. `animacy say` (TTS waveform → motion → robot,
audio in sync) ran on the robot.

## Browser

`web/`: both URDFs; JS retargeter equals the Python one to 1e-6 on 240 random
frames × 2 robots × 2 modes (`tests/test_web_retarget_parity.py`); 240 fps on
an RTX 5080 laptop, 8–12 fps under software rendering.
