# animacy web viewer

A static, no-build browser demo: the **Autonomous Lamp** and the **Reachy
Mini** as animated URDFs side by side, driven by (1) the vendors' own clips,
(2) canonical human-motion clips retargeted in JS through each robot's
`ROBOT.md`, (3) live webcam puppeteering (MediaPipe in the browser), and (4) a
pluggable motion-model slot.

**Talk mode** (the headline demo): type a line → Kokoro-82M TTS runs *in the
browser* → the waveform's audio features → the motion model (or retrieval /
envelope) → canonical frames → both robots, while the voice plays. Motion is
sampled off `AudioContext.currentTime`, so sync is structural.

## Run it

```
# from the repo root
python -m http.server 8000
# → http://localhost:8000/web/
```

Every path is relative, so the same tree serves from GitHub Pages at
`https://<user>.github.io/animacy/web/`. There is no bundler: `index.html`
carries an import map and the browser pulls the pinned dependencies from
jsDelivr / Google storage:

| package | pin | used for |
|---|---|---|
| `three` | 0.170.0 (`build/three.module.js`, `examples/jsm/` as `three/addons/`) | rendering, `OrbitControls`, `STLLoader` |
| `urdf-loader` (gkjohnson) | 0.13.1 (`src/URDFLoader.js`) | URDF → three.js scene graph, `joint.setJointValue` |
| `@mediapipe/tasks-vision` | 0.10.21 (`vision_bundle.mjs` + `wasm/`) | `FaceLandmarker`, `PoseLandmarker`, `DrawingUtils` |
| MediaPipe models | `face_landmarker/float16/1`, `pose_landmarker_lite/float16/1` (storage.googleapis.com) | loaded lazily when Webcam live starts |
| `onnxruntime-web` | 1.20.1 (`dist/ort.min.mjs`, wasm EP single-threaded, `dist/` for the .wasm) | `web/models/a2m_ar.onnx` + `vq_decoder.onnx` (Talk / Listen) |
| `kokoro-js` | 1.2.1 (`dist/kokoro.web.js`, self-contained: bundles transformers.js + its own ORT) | Kokoro-82M-v1.0-ONNX, `q8`, WebGPU → wasm fallback; ~90 MB from huggingface.co, cached by the browser |

Webcam live needs a secure context (`localhost` or https) for `getUserMedia`.

### Layout and the Lamp quick-start row

The page opens in the **hero layout**: the Autonomous Lamp loads first and takes
~62 % of the width (`HERO.weight` in `js/main.js`, applied as the viewports'
`grid-template-columns`), the Reachy Mini sits beside it, and the lamp's camera
is aimed at its head — 3/4 front (`iso`), head ≈ 45 % of the viewport height,
head centre 58 % of the way up (`RobotViewer.frameOnLink`, which projects the
head link's box and corrects distance and target until the measured fill
matches; `animacy.layoutInfo().hero.measured` reports it). The A/B lamp gets the
same framing so A and B match. `?layout=equal` is the plain split (every
viewport 1fr, every robot framed whole) and what the grader's renderer sets up
itself. Extra robots from the picker are 1fr columns.

The **Lamp** row above the source tabs holds three one-click demos (`DEMOS` in
`js/main.js`, also `animacy.demo(name)`):

| button | state it puts the page in |
|---|---|
| Vendor nod vs animacy nod (A/B) | canonical `synth/cal_nod` on both robots + the A/B lamp playing the vendor's `nod` CSV raw |
| Human clip → lamp (look left / brows / lean-in) | a playlist: `cal_look_left_right` → `cal_brows` → `cal_lean_in`, each once, then wraps (`animacy.playlist`) |
| Talk: "No way, that is incredible news!" | Talk tab, backend `retrieval`, voice `af_heart`, intent from the text (excitement), and says it |

## How the pieces connect

```
web/robots/<name>.json          animacy profile export robots/<name>   (joints, retarget modes, description.urdf)
        │
        ├─ description.urdf ──► ../robots/<name>/urdf/*.urdf ──► js/viewer.js  RobotViewer (three + urdf-loader)
        │                        meshes resolve relative to the URDF (../meshes/*.stl)
        └─ retarget.<mode> ────► js/retarget.js  LiveRetargeter  (port of animacy/retarget.py)

motion source (js/motion_source.js)          frame            per robot
  ClipSource     native CSV/JSON ───────────► {joints:{lamp:{…}}} ──► toUrdfValues ──► viewer.setJoints
  ClipSource     canonical clip JSON ───────► {channels:{…}} ─┬──► LiveRetargeter(lamp)  ──► toUrdfValues ──► lamp
  SyntheticSource calibration clips (JS) ───►                 └──► LiveRetargeter(reachy) ──► toUrdfValues ──► reachy
  WebcamSource   MediaPipe → js/canonical.js ► {channels:{…}} (same path, dt from video timestamps)
  TalkSource     text → kokoro-js → js/features.js → js/model.js ► {channels:{…}} clocked to WebAudio
  ListenSource   mic → energy VAD → js/model.js (causal, speaking=0) + gaze overlay from the camera
                 (face position → yaw/pitch target, one-pole 1 Hz, weight 0.5 under the model's head_yaw/pitch;
                  the blend is specified as a comment in js/talk.js so Python can mirror it)  ► {channels:{…}} (experimental)
```

### Talk mode internals (`js/talk.js`, `js/model.js`, `js/features.js`, `js/dsp.js`)

```
text ──kokoro-js──► Float32 @ 24 kHz ──OfflineAudioContext──► 16 kHz
     ──features.js audioFeatures──► [T, 66]   (T = ceil(seconds·30))
     speaking[t] = feats[t, 64] > −0.3        (serve._speaking_from_audio)
     ──backend──►  model:     poolPairs → a2m.onnx logits [L,512] → sampleCodes (T=0.8,
                             bigram prior w=0.5, sfc32 seed) → vq_decoder.onnx [2L,14]
                             → zero-phase Butterworth 6 Hz (dsp.filtfilt == scipy) → hold odd tail
                   retrieval: window keys (330-d) → cosine argmax + continuity/speaking bonus
                             → 5-frame crossfade  (== animacy.model.retrieval.RetrievalIndex.query)
                   envelope:  serve.envelope_motion heuristic (no model; labelled as such)
     ──motionToFrames──► canonical frames (face_valid=1) ──► Track ──► LiveRetargeter × 2
audio ──AudioBufferSourceNode──► speakers; TalkSource.time = ctx.currentTime − startedAt
```

**Intent + post-processing** (so the browser's Talk mode equals `animacy say`):
the typed text is the intent source — `js/intent.js` mirrors
`animacy/model/intent.py` with the lexicon from `model.json` (`intent` block):
tag (greeting / agreement / doubt / excitement / thinking / neutral), arousal,
valence and the amplitude **tier** by tag (excitement 1.45, greeting 1.25,
agreement/doubt 1.15, neutral 1.0, thinking 0.9 — `amplitude_tiers` in the
block); the "intent:" dropdown forces a tag like `--intent`. Retrieval adds to
each window's score 0.15·(1 − |window_arousal − target|), for thinking
0.10·still_then_move, and `proto_weight`·proto[tag][window] — the index's
gesture-prototype score (nod / head-shake / burst / tilt-and-hold / greet,
v2 `proto` fields). Then `postprocessMotion` applies the `postprocess` block in
order: amplitude tier → energy floor (if the utterance's standardised head+brow
RMS is below `energy_floor`, one scalar in [1, 2] lifts the whole utterance) →
pitch floor (0.3 Hz zero-phase baseline of head_pitch not below −3°) →
utterance-final settle (only after speech has ended: linear blend to neutral
over 0.5 s, then hold) → schema clamp.

`web/models/model.json` (written by `animacy/model/export.py`) is the contract:
channel order, `stats`, sampling defaults, smoothing cutoff, file names. The
manifest's `bundle` flags say which files exist; missing ones remove that
backend from the picker and the Talk tab still works on the envelope heuristic.

Two model archs, chosen by `model.json` (`resolveArch` in `model.js`):

| shape | arch | graph | per-step rule |
|---|---|---|---|
| v1: no `archs` key | `ff` | `a2m.onnx` once → logits `[L,512]` | `softmax(logits/T + w·bigram[prev])` (infer.sample_codes) |
| v2: `archs: ["ff","ar"]`, `default_arch` | `ar` | `a2m_ar.onnx` stepped per code with the history `[BOS=512, c0, …]` → `logits_next` | temperature, top-p, repeat penalty, optional stay bias (a2m_ar.generate) |

The default arch is used when its sampler exists and its file is listed;
otherwise the first runnable listed arch (e.g. a v3 `default_arch` this build
does not know falls back to `ar`/`ff`), and if none is runnable
`MotionBackends` drops to retrieval → envelope with a status line — never a
hard error. Only the chosen arch's ONNX is downloaded. The Talk status shows
`model/ff` or `model/ar` and, for `ar`, a per-step progress bar (30 steps ≈ 0.2 s
in wasm on a laptop).

Parity with Python: `tests/test_web_features_parity.py` (features to 1e-4,
`filtfilt` to 1e-9, node) and `tests/test_web_model_parity.py` (headless
Chromium running `model.js` on the real bundle vs onnxruntime + `infer.py` /
`a2m_ar.py`: ff logits < 1e-3, decoder < 1e-3, greedy codes identical for both
archs, full greedy generate < 2e-3, retrieval ids identical, every AR draw
inside Python's top-p nucleus, `sampleTopP` frequencies within sampling noise
of `AudioToMotionAR.sample`; intent analysis identical on 17 lines incl.
negation / caps / ellipsis edge cases; intent-conditioned retrieval ids
identical and post-processed motion < 2e-3 vs `infer.retrieve`). The stochastic samplers use sfc32 rather than
numpy's PCG64, so seeded sequences are reproducible in the browser but not
bit-identical to Python; the tests score the JS draws under Python's per-step
distributions instead.

* `js/main.js` — boot, UI, the single `requestAnimationFrame` loop, and the
  `window.animacy` API used by the dev scripts (`setSource`, `setClip`,
  `setMode`, `seek`, `getJointValues`, `getChannels`, `setView`, …).
* `js/clips.js` — parsers for Autonomous OS CSV (`timestamp,<joint>.pos`, 20 Hz),
  animacy joint-table JSON, canonical clip JSON, plus the synthetic calibration
  clips and a `Track` that interpolates any column by time.
* `js/canonical.js` — the channel table (mirrors `animacy/schema.py`) and the
  MediaPipe → canonical derivation with every sign written out against
  `docs/CANONICAL.md`; `NeutralCalibrator` (median of the last second).
* `js/standins.js` — dev-only fallback URDFs used when a robot's real URDF is
  missing (`?standin=1` forces them; `?urdf_lamp=<url>` / `?urdf_reachy_mini=<url>`
  point at any URDF). The LeLamp stand-in is GPL reference material and is
  never the shipped default.
* `manifest.json` — what exists on disk (URDFs, native clips, captured clips,
  `web/models/*.onnx`), written by `python web/dev/build_manifest.py`, so the
  static site never probes for files that would 404. Re-run it after adding
  clips, URDFs or models; on a local `http.server` new files in `web/clips/`
  are picked up anyway. The **Model** tab reads `manifest.models`: with no
  model present it says "coming soon"; with `a2m_ar.onnx`/`vq_decoder.onnx`
  present it names them. Nothing is fetched until a runner is wired
  (`ModelSource` in `js/motion_source.js`; the bundle's contract is in
  `web/models/model.json`, written by `animacy/model/export.py`).

URL parameters: `?source=native|canonical|webcam|model`, `?clip=<id>`
(e.g. `lamp/nod`, `synth/cal_look_left_right`, `clip/<name>`), `?mode=default|puppet`,
`?ab=1`, `?autoplay=0`.

### Coordinate frames

URDFs are z-up; three.js is y-up. The robot root sits under a group rotated
−90° about x, so URDF +x (forward) stays +x, URDF +z becomes +y (up) and
URDF +y (robot's left) becomes three −z. "Front" view (`animacy.setView('front')`)
puts the camera on +x: the robot's left is on the viewer's right, which is the
quickest way to eyeball a sign. `animacy.linkForward(robot, 'head')` returns a
link's world forward axis so a sign can be *measured*: `screenshot.py` requires
both heads to swing to −z on the "look LEFT" calibration clip and to +y on
"look UP", whatever the gains in `ROBOT.md` are.

## The JS retargeter mirrors the Python one

`js/retarget.js` is a line-for-line port of `animacy/retarget.py`:

```
LiveRetargeter.step(channels, dt):           # docs/RETARGET.md
  x      = Σ gain·channel                     (missing / NaN channel → 0)
  x      = 0 if |x| < deadband else x − sign(x)·deadband
  u      = softClip(x + offset + rest, soft_limit)   # tanh knee over the last soft_limit of the range
  u      = clamp(u, [min, max])               # mapping bounds, else joint limits
  u      = clamp(u + gate·idleValue(clock))   # idle sway, gated off while the target moves
  y,v    = spring ? springStep(prev, vel, u)  # exact zero-order-hold damped oscillator (springCoefficients)
         : prev + (1 − exp(−2π·cutoff·dt))·(u − prev), null   # one-pole, cutoff = smooth_hz or 6 Hz
  y,v    = clipStep(prev, y, v)              # |Δ| ≤ max_speed·dt, clamp [joint.min, joint.max];
                                             # carry the spring's own v unless a limit engaged, then (y − prev)/dt
toUrdfValues: (value + urdf_offset)·urdf_sign, deg→rad, mm→m, keyed by urdf_joint
```

`tests/test_web_retarget_parity.py` runs both implementations on the same
random channel stream (via `node web/dev/retarget_parity.mjs`) and requires
agreement to 1e-6 — on the shipped profiles and on a synthetic v1.1 profile
that uses `soft_limit`, `idle` and `spring`; it also checks that
`web/robots/*.json` and `manifest.json` are current. Clip playback hands the retargeter *clip-time* `dt` (so 2× speed
is a faster robot, as it would be on hardware); the webcam hands it wall-clock
`dt` from the video frame timestamps. `main.js` never feeds the retargeter
more than one nominal 1/30 s step: longer gaps are split into equal
sub-steps (the exact spring composes, so this changes nothing but keeps the
rate limit per-frame), and a scrub (`frame.seek`) is settled with 30 nominal
steps instead of one giant one.

## Record mode (contribute data from the page)

In **Webcam live**: `subject` + role (speaking / listening) → **● Record** /
**■ Stop** → **Download take** gives `<subject>_<slug>.zip` containing
`motion.json` (canonical clip JSON, 30 Hz, `null` where the face was not
tracked), `audio.webm` (MediaRecorder opus, t = 0 at the first audio sample)
and `meta.json` (`source: webcam-browser`, role, arm, the raw neutral pose,
`license: CC-BY-4.0`, tool versions). **Guided session** walks through the
same prompts as `scripts/record_me.py` (read aloud with the Web Speech API,
3-2-1 countdown, timed take) and bundles every take into one zip. Import with
`animacy import-browser <zip> -o data/clips/<name>`. Listening takes get
`speaking = 0` (the microphone hears the podcast, not the contributor).

## Adding a robot's viewer entry

No JavaScript. The viewer's robot set is `web/manifest.json`:

1. `animacy check robots/<name>` passes (URDF + meshes in `robots/<name>/`).
2. `animacy profile export robots/<name> -o web/robots/<name>.json`.
3. `python web/dev/build_manifest.py` — the robot now appears in the
   **+ add robot** picker (header) and opens as an extra viewport driven by
   the same source; `?robots=lamp,reachy_mini,<name>` opens it at load. The
   headline pair (`HEADLINE_ROBOTS` in `js/main.js`: lamp, reachy_mini) is the
   default layout. Retarget modes in the picker are the union over loaded
   robots (a robot without the chosen mode uses its `default`). Native clips:
   point `native_clips.dir` in `ROBOT.md` at a folder of Autonomous CSV or
   animacy joint-table JSON files (an `index.json` with
   `{"clips":[{"name","description"}]}` adds descriptions) — they show up in
   the Native clip list by extension.
4. `python web/dev/screenshot.py` (it adds every non-headline manifest robot,
   plays the puppet wave on it and removes it again).

Only `description.viewer.camera_distance` is read from the profile for
framing; the viewer never moves the camera closer than what fits the robot's
bounding box.

## Verification

```
python web/dev/screenshot.py        # headless: zero console errors, both robots load, native/canonical/
                                    # puppet/A-B/captured/talk(backends)/listen/webcam checks, PNGs in web/dev/shots/
python web/dev/screenshot.py --with-tts   # + Kokoro TTS in the page (downloads ~90 MB once)
python web/dev/verify_pages.py      # the LIVE GitHub Pages deployment: URDFs/meshes/clips over Pages, zero errors
python web/dev/fps.py               # headed: FPS on the machine's GPU
python web/dev/probe.py "<js expr>" # evaluate anything against the live page
python -m pytest tests/test_web_retarget_parity.py tests/test_web_features_parity.py tests/test_web_model_parity.py
```

Webcam mode is exercised with Chromium's fake camera (`--use-fake-device-for-media-stream`)
to prove it initialises without throwing; the face/pose derivation itself is
only verifiable with a person in front of a real camera (use the calibration
clips as the reference for which way each channel should move).

### Demo video (`web/dev/demo_video.py`)

```
python web/dev/demo_video.py                 # docs/media/animacy_lamp_60s.mp4 (+ lamp_nod_ab_12s.mp4, lamp_nod_ab.gif)
python web/dev/demo_video.py --headed --gpu  # Kokoro on WebGPU (faster TTS), same output
python web/dev/demo_video.py --no-tts        # synthetic placeholder voice; the captions say so
```

Nothing is screen-recorded. The page is parked (`animacy.setCapture(true)`
stops the rAF loop and hides the fps pill) and advanced one frame at a time
(`animacy.stepFrame(1/30)` runs one tick: source → retargeters → render →
readouts), and every frame is a screenshot, so frame *i* is exactly *i*/30 s of
clip time however slow the renderer is. In capture mode `TalkSource` runs on a
manual clock (`manualClock`): the real Kokoro voice is synthesised in the page,
the waveform is pulled out (`talk.lastAudio`) and muxed at the frame the line
started, so voice and motion share one clock by construction — the same idea as
the Python runtime and the grader's renderer. Captions and the end card are
ffmpeg `drawtext`; the shot list is in `docs/SUBMISSION.md`.
