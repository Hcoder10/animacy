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

## v2a (2026-08-27): autoregressive decoder, 39 clips, two held-out speakers

`checkpoints/v2a/REPORT.md` (command recorded there; `python -m animacy.model.train
--arch both --holdout kende_interview_2014 obama_2015_02_07 --exclude
sd_rapper_interview cbp_vlog_day2 --speaker-cap 0.6 --ar-change-weight 4 ...`).
Data at run start: 39 license-verified clips, 152 valid minutes, 37 training clips
(145.8 effective minutes; the 60 % per-speaker cap did not bind). Mirror + time-warp
augmentation on the training set only. Both held-out speakers were never seen.

**What changed from v1.** (a) `a2m_ar`: an autoregressive code decoder (causal
self-attention over the code history, window 32, cross-attention into the same
audio trunk; listen mode is causal in both) with a change-weighted loss (x4 on
code transitions, because 43 % of code steps repeat the previous code and plain
CE is minimised by freezing) and early stopping; (b) a fresh tokenizer on the
larger corpus (509/512 codes used, val MAE 0.23 vs 0.62 predict-mean);
(c) sampling defaults picked by the mean over both hold-outs (T = 1.0, top-p 0.9,
repeat penalty 1.0); (d) a promotion rule applied per speaker: a learned
generator ships as the default only if on **every** held-out speaker it beats its
own shuffled-audio control by >= 0.05 head-beat recall, sits within 0.05 of the
ground-truth stillness, and its code NLL is below the unigram floor.

| held-out speaker | generator | code NLL (floor) | head-beat recall vs shuffled (margin) | stillness (truth) | velocity W1 rel. |
|---|---|---|---|---|---|
| obama_2015 | v1 ff | 6.031 (6.167) | 0.617 / 0.497 (+0.12) | 0.025 (0.103) | 0.30 |
| obama_2015 | v1-AR (`v1ar_holdout_obama2015`) | 3.578 (5.847) | 0.452 / 0.359 (+0.09) at T1.0/p0.9 | 0.116 (0.103) | 0.35 |
| obama_2015 | **v2a ff** | 5.211 (5.663) | 0.403 / 0.403 (0.00) | 0.105 (0.103) | 0.23 |
| obama_2015 | **v2a AR** | **3.557** (5.663) | 0.438 / 0.497 (-0.06) | 0.084 (0.103) | **0.19** |
| obama_2015 | retrieval (v2a index) | - | 0.538 / 0.486 (+0.05) | 0.140 (0.103) | 0.19 |
| kende | v1 ff | 6.276 (6.162) | 0.644 / 0.613 (+0.03) | 0.013 (0.097) | 1.94 |
| kende | v1-AR (`v1ar`) | 4.280 (6.064) | 0.545 / 0.506 (+0.04) at T1.0/p1.0 | 0.044 (0.097) | 1.58 |
| kende | **v2a ff** | 5.548 (5.481) | 0.253 / 0.273 (-0.02) | 0.145 (0.097) | 0.95 |
| kende | **v2a AR** | **3.603** (5.481) | 0.391 / 0.403 (-0.01) | 0.090 (0.097) | 1.07 |
| kende | retrieval (v2a index) | - | 0.470 / 0.510 (-0.04) | 0.139 (0.097) | 1.21 |

Pooled over both speakers (546 ground-truth head peaks): AR NLL 3.581 vs floor
5.566; head-beat recall AR 0.418 vs 0.469 shuffled, ff 0.339 vs 0.355, retrieval
0.509 vs 0.467. Retarget legality: 0 speed-cap violations for every source on
both robots. The 40 %-cap variant (`checkpoints/v2_interim`, run on 36 clips)
gives the same picture: AR NLL 3.947 vs floor 5.613, AR 0.377 vs 0.363 shuffled,
retrieval 0.500 vs 0.469.

**Reading, stated plainly.** The autoregressive model fixes what v1 could not:
its code NLL is ~2 nats below the unigram floor on *both* unseen speakers, its
velocity statistics are the closest to human of any source (W1 0.19 on obama),
and its stillness matches the ground truth to within 0.02 - it no longer jitters
and no longer freezes. What it does **not** show is beat alignment: on neither
speaker does any learned generator beat its own shuffled-audio control by the
required 0.05 (the best cells of the sampling sweep reach +0.03 to +0.04 on the
mean and none holds on both speakers), so **we make no claim that the learned
motion is timed to the speech beyond what a shuffled soundtrack gives**.
Retrieval - guaranteed human motion, aligned by construction - therefore
remains the default source (`web/models/model.json: default_backend =
retrieval`); the AR model is exported and selectable (`default_arch = ar`).
The promotion rule stays in the trainer so a future run flips the default only
on evidence. Sampling-time knobs (repeat penalty, stay bias) move the stillness
anywhere between 0.03 and 0.3 without moving the margin.
