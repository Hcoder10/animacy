# Grant submission kit — Autonomous Open Source Grants, Week 5 (Autonomous Lamp, closes Aug 28 2026)

Where to submit: reply with the repo under Dee's post
https://x.com/dee_hw/status/2090803311784145059 ("Drop your repo below and take
him home for free") and in the r/coolgithubprojects thread "Drop your repo and
win an Autonomous Lamp". Past winners were working, useful open source
(vdo.ninja, Node.js libraries, hermes-concurrent-agents, Neon Vision Editor).
Every sentence below is backed by a file in this repo — see the table at the
end; nothing here goes beyond `docs/RESULTS.md` and the evidence files.

Links: repo https://github.com/Hcoder10/animacy · demo https://hcoder10.github.io/animacy/web/

---

## 1. X reply (single post, 276 characters)

```
animacy — the open interaction layer for expressive robots: human motion in, any robot out, one ROBOT.md per body. Runs on a real Reachy Mini today and writes the Lamp's own /servo/upload CSV. Apache-2.0.
Demo: hcoder10.github.io/animacy/web/
Repo: github.com/Hcoder10/animacy
```

### Thread variant (reply to your own post, 3 tweets)

**1/3** (276)
```
animacy — the open interaction layer for expressive robots: human motion in, any robot out, one ROBOT.md per body. Runs on a real Reachy Mini today and writes the Lamp's own /servo/upload CSV. Apache-2.0.
Demo: hcoder10.github.io/animacy/web/
Repo: github.com/Hcoder10/animacy
```

**2/3** (280)
```
What is verified: Reachy Mini sim-to-real on a physical unit (every axis read back within a few degrees, owner confirmed directions); JS retargeter = Python to 1e-6; 0 speed-cap violations on every clip; Lamp URDF built from the vendor CAD and checked against all 31 vendor clips.
```

**3/3** (275)
```
Not yet: no Lamp on hand, so the Lamp path is checked against Autonomous's own route code, not hardware. The learned speech->motion model beats its floors on one held-out speaker, not the other, so retrieval ships as default. Numbers and gaps: docs/RESULTS.md, README Status.
```

---

## 2. Reddit comment (r/coolgithubprojects, ~230 words)

```
**animacy — the open interaction layer for expressive robots** (Apache-2.0)
Repo: https://github.com/Hcoder10/animacy · Demo in the browser: https://hcoder10.github.io/animacy/web/

Human motion in, any robot's motion out. One canonical human-motion space captured from any video or webcam with MediaPipe, and one `ROBOT.md` per robot: joints, limits, safety ceiling, a linear mapping from those channels. Adding a robot is one file; no retraining. The demo plays the Autonomous Lamp and the Reachy Mini side by side on their real URDFs: vendor clips, retargeted human clips, your webcam live, and a talk mode where the robot speaks (TTS in the page) and moves in sync.

**Verified**
- Reachy Mini on a physical unit: clip → `ROBOT.md` → daemon at 30 Hz; every axis read back within a few degrees; owner confirmed all five directions.
- Browser retargeter equals the Python one to 1e-6; 0 speed-cap violations on every clip.
- Lamp URDF from Autonomous's own CAD, directions checked against their device notes and all 31 vendor recordings; the exporter writes the CSV `/servo/upload` accepts.
- 177 tests; two LeRobot v3.0 datasets on the Hub.

**Not verified / not done**
- No Lamp on hand: the Lamp path is checked against Autonomous's route code, not hardware.
- The learned speech→motion model beats its floors on one held-out speaker, not the other; retrieval is the default. The blind-grader gate has not run yet.

It targets four items in Autonomous OS's own `docs/not-built-yet.md` (cross-body moves, community moves for Reachy, recordings under the safety gate, a live policy behind `/policy/run`); see the README.
```

Word count: about 250 (excluding the two link lines).

---

## 3. Demo video — 60 seconds, shot list

Target: 1920×1080, 30 fps, six shots, hard cuts, one caption per shot (bottom
third, plain sans-serif on a dark band), no music over the talk-mode audio.
Record the screen with OBS or Windows Game Bar (`Win+Alt+R`), the robot with a
phone in landscape. Serve the viewer locally so the webcam shot has a secure
context: `python -m http.server 8000` from the repo root → `http://localhost:8000/web/`.
Open the browser console (F12) once; every viewer shot below has a one-line
`window.animacy.*` command that puts the page in the exact state, so takes are
repeatable.

| t | shot | how to produce it | caption |
|---|---|---|---|
| 0–6 s | **Title over the viewer loading both robots.** | `http://localhost:8000/web/?source=native&clip=lamp/nod` — both URDFs load, the Lamp plays its vendor `nod`. Title card text: "animacy — human motion in, any robot out". | *One human motion space. One ROBOT.md per body.* |
| 6–16 s | **Vendor clip A/B.** Left: a human nod retargeted through `ROBOT.md`; middle: the vendor's hand-authored `nod`; right: the same human nod on Reachy. | `?source=canonical&clip=synth/cal_nod&ab=1`, then in the console: `await window.animacy.setAbClip('lamp/nod'); window.animacy.play()`. (Reference frame: `web/dev/shots/07_ab_nod_vs_vendor_nod.png`.) | *A human nod, retargeted (left) next to the Lamp's own nod (middle). Same clip on Reachy (right).* |
| 16–26 s | **Look-left calibration, front view, both robots.** Camera on +x, so the robots' left is the viewer's right. Both heads swing the same way. | `?source=canonical&clip=synth/cal_look_left_right`, console: `window.animacy.setView('front'); window.animacy.play()`. Then `window.animacy.setClip('synth/cal_look_up_down')` for a 3-s tail of both heads tipping up. | *Signs live in ROBOT.md, never in the data. "Look left" turns both bodies to their left.* |
| 26–36 s | **Webcam live.** You in the corner (the page's own preview), both robots following your head turns, a brow raise → antennas / lamp head tip. | Webcam tab, or `?source=webcam`. Hold still 1 s for the neutral pose, then: turn left, turn right, look up, raise brows, lean in. | *Live from a webcam: MediaPipe → canonical channels → each robot's mapping, in the browser.* |
| 36–48 s | **Talk mode.** Type the line, pick the voice, "Say it": the voice plays and both robots move to it. | Talk tab (`?source=talk`), text: `Hi, I'm animacy. One motion space for any expressive robot.` Voice `af_heart`, source `retrieval (motion matching, default)`. First run downloads Kokoro (~90 MB) — do a warm-up take before recording. Keep the page audio in the recording. | *Text → TTS in the page → speech features → motion → both robots, in sync. Retrieval is the default source; the learned model is selectable.* |
| 48–60 s | **The real Reachy Mini (phone video).** Two beats: it speaks and moves; then it mirrors a public-domain clip playing on the laptop next to it. | Beat 1 (6 s): `animacy say "Hello! I'm animacy, running on a real Reachy Mini." --robot reachy_mini --url http://192.168.1.60:8000 --source retrieval --checkpoint checkpoints/v1` (laptop speaker audible). Beat 2 (6 s): `animacy mirror --source "data/raw/2013_09_14_President_Obama_s_Weekly_Address.webm" --robot reachy_mini --sink reachy_daemon --url http://192.168.1.60:8000 --preview --start 30 --duration 20` with the `--preview` window visible on the laptop in frame. | Beat 1: *Same pipeline, real robot: `animacy say`, audio and motion from one waveform.* Beat 2: *`animacy mirror`: a public-domain weekly address (White House, 2013) driving the robot live at 30 Hz.* |
| last 3 s | **End card.** | Black card, two lines. | *github.com/Hcoder10/animacy · Apache-2.0. Not yet: no Lamp on hand; the learned model ships behind retrieval. Details in README → Status.* |

Notes for the operator:

- Reachy: power on, wait ~1–2 min for the daemon, then the commands above enable motors and `wake_up` themselves (`animacy.sinks.ReachyDaemonSink.prepare`). Frame the head and antennas; the body yaw is visible on big turns.
- `--source retrieval` needs `checkpoints/v1/retrieval.json` (present on the dev machine, not in git). `--source envelope` needs nothing and is a labelled heuristic — if used, the caption must say "envelope heuristic", not "model".
- The `mirror` preview window shows the tracked face; keep it in the phone shot so the viewer sees source and robot together. The PD source and its license record are in `data/raw/sources.json` and the dataset card.
- Keep the fps counter visible in the viewer shots (top right); on the dev laptop it reads ~240 fps.

Assembly (one command per stage; `ffmpeg` on PATH):

```bash
# 1. trim each take to its slot (example: the A/B shot, 10 s from 00:00:04)
ffmpeg -ss 4 -t 10 -i take_ab.mp4 -vf "scale=1920:1080,drawtext=fontfile=C\\:/Windows/Fonts/segoeui.ttf:text='A human nod, retargeted (left), next to the Lamp\\'s own nod (middle). Same clip on Reachy (right).':fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=14:x=(w-text_w)/2:y=h-120" -r 30 -c:v libx264 -crf 18 -c:a aac shot2.mp4
# 2. list the shots in order
printf "file 'shot1.mp4'\nfile 'shot2.mp4'\nfile 'shot3.mp4'\nfile 'shot4.mp4'\nfile 'shot5.mp4'\nfile 'shot6.mp4'\nfile 'end.mp4'\n" > shots.txt
# 3. concatenate (re-encode so mixed sources cut cleanly)
ffmpeg -f concat -safe 0 -i shots.txt -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -movflags +faststart animacy_demo_60s.mp4
```

Optional, for a clean silent render of any joint table on the real URDFs
(same renderer the blind grader uses): `animacy/grade/render.py` drives the
viewer headlessly with Playwright and encodes 30 fps H.264 — no screen
recording needed for shots 1–3 if you prefer.

---

## 4. Claims → backing files

| claim (as worded above) | backing file(s) |
|---|---|
| "the open interaction layer for expressive robots: human motion in, any robot out, one ROBOT.md per body" | `README.md`, `docs/CANONICAL.md`, `docs/ROBOT_MD_SPEC.md`, `robots/*/ROBOT.md` |
| 28 channels at 30 Hz: head 6-DoF, gaze, brows, mouth, torso, an arm | `docs/CANONICAL.md` (channel table), `animacy/schema.py` |
| captured from any video or webcam with MediaPipe | `animacy/capture.py` (module docstring: sources, models), `animacy/mirror.py` |
| adding a robot is one file; no retraining | `docs/ADD_A_ROBOT.md`, `docs/ROBOT_MD_SPEC.md` rule 1, `robots/_template/ROBOT.md`; `robots/so101/` added by following the doc (commit fdddaae) |
| demo plays the Lamp and Reachy Mini side by side on their real URDFs; vendor clips, retargeted clips, webcam, talk mode with TTS in the page | `web/README.md` (modes, dependency pins, Kokoro-js), `web/dev/shots/*.png`, `robots/lamp/urdf/lamp.urdf`, `robots/reachy_mini/urdf/reachy_mini.urdf` |
| "Runs on a real Reachy Mini today" / every axis read back within a few degrees / owner confirmed all five directions | `docs/evidence/reachy_sim2real_20260826.md` (table + "Visual confirmation … CONFIRMED"), raw log `docs/evidence/reachy_sim2real_20260826_214727.json`, `scripts/reachy_sim2real.py` |
| `animacy say` ran on the robot, audio in sync | `docs/RESULTS.md` § Sim-to-real; `animacy/serve.py` |
| "writes the Lamp's own /servo/upload CSV" / "exactly the CSV `/servo/upload` accepts" | `animacy/export.py` (`to_autonomous_os_csv`, `validate_autonomous_os_csv` mirrors `hal/routes/servo.py`), `docs/AUTONOMOUS_OS.md` |
| JS retargeter = Python to 1e-6 | `tests/test_web_retarget_parity.py`, `web/README.md` § "The JS retargeter mirrors the Python one", `docs/RESULTS.md` § Browser |
| 0 speed-cap violations on every clip | `docs/RESULTS.md` (retarget legality row), `animacy/retarget.py` (`stretch_timeline`, `rate_limit`) |
| Lamp URDF derived from Autonomous's CAD; directions checked against their device notes and all 31 vendor recordings | `robots/lamp/urdf/README.md`, `robots/lamp/ROBOT.md` § Sign conventions, `tests/test_lamp_urdf.py`, `robots/lamp/urdf/preview/contact_sheet.png`, `robots/lamp/clips/native/` (31 files) |
| 177 tests | `tests/` (`pytest --co -q` on commit 3dcaf82 → 177 collected) |
| two LeRobot v3.0 datasets on the Hub validated with the real loader | `docs/LEROBOT.md` (repo ids, validator), `scripts/export_lerobot.py`, `tests/test_lerobot_export.py` |
| no Lamp on hand; Lamp path checked against route code, not hardware | `robots/lamp/ROBOT.md` ("verified in simulation only"), `animacy/sinks.py` (`AutonomousHalSink` docstring: UNVERIFIED), `README.md` § Status |
| the learned model beats its floors on one held-out speaker, not the other; retrieval is the default | `docs/RESULTS.md` § Audio → motion (kende vs obama_2015 columns; "Verdict"), `web/models/model.json` (`default_backend: retrieval`) |
| the blind-grader acceptance gate has not run yet | `animacy/grade/__init__.py` (gate definition), `README.md` § Results ("pending first run") |
| targets four items in Autonomous OS's `docs/not-built-yet.md` | `README.md` § For Autonomous OS (each ask quoted with the command), `docs/AUTONOMOUS_OS.md` |
| Apache-2.0 | `LICENSE`, `pyproject.toml` (`license = Apache-2.0`), `README.md` § License and attributions |
| public-domain clip used in the mirror shot | `data/raw/sources.json` (license record for the 2013-09-14 weekly address), HF dataset card `squaredcuber/animacy-human-motion` |
| ~240 fps on the dev laptop | `docs/RESULTS.md` § Browser (240 fps on an RTX 5080 laptop), `web/dev/fps.py` |
