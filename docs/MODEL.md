# The motion model and the interaction layer

## What "VLA" means here

The robots animacy targets do not manipulate; they *behave*. The action space
is expressive motion, the "language" is what the robot is saying or hearing,
and "vision" is who it is looking at. So the model is:

```
speech audio (what the robot says / hears) + role (speaking | listening) + [text]
        ──▶  canonical human motion at 30 Hz  ──▶  ROBOT.md retarget  ──▶  body
gaze target from the camera (where the person is)  ──▶  overlay on head_yaw/pitch
```

One model, in the **canonical human space** (`docs/CANONICAL.md`), so every
`ROBOT.md` body gets it without retraining. The same clips export to LeRobot
datasets with text labels (`animacy export lerobot`), which is how the data
feeds proper VLA training later.

## Why speech-driven

Human motion in conversation is overwhelmingly structured by speech: nods at
phrase boundaries, brow raises on stress, lean-ins on questions, stillness
while listening. Conditioning on the audio the robot is about to play gives
**synchrony for free** — the motion is generated from the same waveform that
comes out of the speaker. Text semantics ride on prosody, and an optional text
head (the reachy-duplex ramps recipe) adds explicit intent later.

The training pairs are simply **people talking on video**: the audio track and
the captured canonical motion of the same person. That is why `capture` matters
more than the model.

## Architecture (v1 — small on purpose; must run in the browser)

1. **Audio features** (`animacy/features.py`, mirrored in `web/js/features.js`):
   16 kHz mono → 64 log-mel at 100 Hz (win 25 ms, hop 10 ms) → averaged onto the
   30 Hz motion grid → + log energy + Δenergy → per-utterance mean/var norm.
   66 dims per tick. TTS and real microphones both normalise into the same space.
2. **Motion tokenizer** (`animacy/model/vq.py`): VQ-VAE over 8-frame windows of
   the 14 *model channels* (head 6, brow 3, torso 3, mouth_open, smile), one code
   per 2 frames (15 codes/s), codebook 512 × 64, EMA updates and **dead-code
   revival** (without revival a 512 codebook collapses to a handful of entries
   while the loss keeps falling — measured on reachy-duplex). Codes are
   interpretable primitives: nods, tilts, turns, brow raises, settles.
3. **Audio → codes** (`animacy/model/a2m.py`): a non-causal Transformer encoder
   (d 192, 4 layers) over the 15 Hz feature stream + a `speaking` flag,
   predicting a code distribution per step. Non-causal is fine for *talk* mode
   because the utterance audio is known before playback; for *listen* mode the
   model sees only the past (a causal mask is a flag). Inference = temperature
   sampling with a bigram transition prior (learned code→code counts) so
   sequences stay coherent without an autoregressive decoder in ONNX.
4. **Decode** → VQ decoder → 30 Hz motion → zero-phase smoothing per utterance
   → canonical frames → `LiveRetargeter` per robot.
5. **Retrieval baseline** (`animacy/model/retrieval.py`): motion matching. Index
   1 s windows of (audio features → human motion) from the corpus; at run time
   pick the nearest window per 0.5 s hop with a continuity bias, crossfade. It
   is guaranteed to be *human* motion and *aligned* to speech; it is the floor
   the learned model must beat on the held-out metrics, and the fallback the
   demo ships if it does not.

Both run in the browser: ONNX Runtime Web for (3)+(4) (< 10 MB), plain JS for
(5). Kokoro-js (82M, WebGPU/wasm) synthesises the robot's speech client-side and
hands back the waveform, so **the whole talk loop is local in the page**.

## Modes

| mode | input | who moves | source |
|---|---|---|---|
| talk | text → TTS waveform | the robot, as speaker | model (speaking=1) |
| listen | microphone | the robot, as listener | model (speaking=0, causal) + gaze overlay |
| mirror | webcam | the robot copies you | capture → retarget (no model) |
| clip | a captured or vendor clip | playback | file |

Role matters: a listener and a speaker move differently, and a model trained
without the flag learns listening-as-speaking (measured on reachy-duplex).

## Evaluation (held-out speakers)

- code accuracy / NLL vs the retrieval baseline and a "majority code" floor;
- motion statistics: per-channel velocity histograms and stillness ratio vs
  ground truth (a model that collapses to idle fails this even with good NLL);
- beat alignment: fraction of ground-truth motion peaks within ±150 ms of a
  generated peak, and vs shuffled audio (the model must beat its own shuffle);
- retarget legality: 0 speed-cap violations after `retarget_clip`.

Numbers go in `docs/RESULTS.md` with the exact commands; nothing is claimed as
measured without them.

## Data

- `scripts/fetch_sources.py`: public-domain / CC-BY talking-head video only,
  license verified from metadata, ND refused. Sources and evidence in
  `data/raw/sources.json`; the HF dataset card repeats them.
- Your own webcam (`animacy capture --source 0`): the most valuable data for a
  desk robot is a person at a desk talking to a camera.
- Reachy's 85 Pollen emotions and the Lamp's 31 clips are *not* training data
  for the model (they are robot-space, hand-authored); they are the envelope the
  retarget is tuned to and the A/B in the viewer.

## Files

```
animacy/features.py          audio features (numpy)      ↔ web/js/features.js
animacy/model/vq.py          tokenizer                   → web: vq_decoder.onnx
animacy/model/a2m.py         audio→codes                 → web: a2m.onnx
animacy/model/retrieval.py   motion matching             ↔ web/js/retrieval.js
animacy/model/data.py        clips → training tensors
animacy/model/train.py       train vq, then a2m; writes checkpoints + metrics
animacy/model/export.py      ONNX + retrieval index for the web
animacy/serve.py             the interaction runtime for a real robot
```
