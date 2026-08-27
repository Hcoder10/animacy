# animacy web viewer

A static, no-build browser demo: the **Autonomous Lamp** and the **Reachy
Mini** as animated URDFs side by side, driven by (1) the vendors' own clips,
(2) canonical human-motion clips retargeted in JS through each robot's
`ROBOT.md`, (3) live webcam puppeteering (MediaPipe in the browser), and (4) a
pluggable motion-model slot.

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

Webcam live needs a secure context (`localhost` or https) for `getUserMedia`.

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
  ModelSource    ONNX (stub, contract in the file)
```

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
* `manifest.json` — what exists on disk (URDFs, native clips, captured clips),
  written by `python web/dev/build_manifest.py`, so the static site never
  probes for files that would 404. Re-run it after adding clips or URDFs; on a
  local `http.server` new files in `web/clips/` are picked up anyway.

URL parameters: `?source=native|canonical|webcam|model`, `?clip=<id>`
(e.g. `lamp/nod`, `synth/cal_look_left_right`, `clip/<name>`), `?mode=default|puppet`,
`?ab=1`, `?autoplay=0`.

### Coordinate frames

URDFs are z-up; three.js is y-up. The robot root sits under a group rotated
−90° about x, so URDF +x (forward) stays +x, URDF +z becomes +y (up) and
URDF +y (robot's left) becomes three −z. "Front" view (`animacy.setView('front')`)
puts the camera on +x: the robot's left is on the viewer's right, which is the
quickest way to eyeball a sign.

## The JS retargeter mirrors the Python one

`js/retarget.js` is a line-for-line port of `animacy/retarget.py`:

```
LiveRetargeter.step(channels, dt):
  x      = Σ gain·channel                     (missing / NaN channel → 0)
  x      = 0 if |x| < deadband else x − sign(x)·deadband
  target = clamp(x + offset + rest, [min, max])   # mapping bounds, else joint limits
  alpha  = 1 − exp(−2π·cutoff·dt)             # cutoff = smooth_hz or the 6 Hz default
  y      = prev + alpha·(target − prev)
  y      = prev + clamp(y − prev, ±max_speed·dt)
  y      = clamp(y, [joint.min, joint.max])
toUrdfValues: (value + urdf_offset)·urdf_sign, deg→rad, mm→m, keyed by urdf_joint
```

`tests/test_web_retarget_parity.py` runs both implementations on the same
random channel stream (via `node web/dev/retarget_parity.mjs`) and requires
agreement to 1e-6; it also checks that `web/robots/*.json` and `manifest.json`
are current. Clip playback hands the retargeter *clip-time* `dt` (so 2× speed
is a faster robot, as it would be on hardware); the webcam hands it wall-clock
`dt` from the video frame timestamps.

## Adding a robot's viewer entry

1. `animacy check robots/<name>` passes (URDF + meshes in `robots/<name>/`).
2. `animacy profile export robots/<name> -o web/robots/<name>.json`.
3. Add the name to `ROBOT_NAMES` in `js/main.js` and a `<section class="viewport" id="vp-<name>">`
   (with `badge-`, `sub-`, `loading-` children) plus a `joint-bars-<name>` panel in `index.html`.
   Native clips: point `native_clips.dir` in `ROBOT.md` at a folder of
   Autonomous CSV or animacy joint-table JSON files (an `index.json` with
   `{"clips":[{"name","description"}]}` adds descriptions).
4. `python web/dev/build_manifest.py`, then `python web/dev/screenshot.py`.

Only `description.viewer.camera_distance` is read from the profile for
framing; the viewer never moves the camera closer than what fits the robot's
bounding box.

## Verification

```
python web/dev/screenshot.py        # headless: zero console errors, both robots load,
                                    # native/canonical/puppet/A-B/webcam checks, PNGs in web/dev/shots/
python web/dev/fps.py               # headed: FPS on the machine's GPU
python web/dev/probe.py "<js expr>" # evaluate anything against the live page
python -m pytest tests/test_web_retarget_parity.py
```

Webcam mode is exercised with Chromium's fake camera (`--use-fake-device-for-media-capture`)
to prove it initialises without throwing; the face/pose derivation itself is
only verifiable with a person in front of a real camera (use the calibration
clips as the reference for which way each channel should move).
