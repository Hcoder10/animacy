# Grant submission kit — Autonomous Open Source Grants, Week 5 (Autonomous Lamp, closes Aug 28 2026)

Where to submit: reply with the repo under Dee's post
https://x.com/dee_hw/status/2090803311784145059 ("Drop your repo below and take
him home for free") and in the r/coolgithubprojects thread "Drop your repo and
win an Autonomous Lamp". Past winners were working, useful open source
(vdo.ninja, Node.js libraries, hermes-concurrent-agents, Neon Vision Editor).
Every sentence below is backed by a file in this repo — see the table at the
end; nothing here goes beyond `docs/RESULTS.md` and the evidence files.

Links: repo https://github.com/Hcoder10/animacy · demo https://hcoder10.github.io/animacy/web/
· demo video
https://github.com/Hcoder10/animacy/releases/download/v0.1.0/animacy_demo.mp4
— the 1080p master ships as a release asset because
`docs/media/animacy_demo.mp4` is deliberately gitignored at 39.6 MB, so its
`raw/main` URL 404s. The 720p cut is tracked and does resolve:
`https://github.com/Hcoder10/animacy/raw/main/docs/media/animacy_demo_720.mp4`.
Replace every `<VIDEO_URL>` below with whichever you post (a YouTube unlisted
link is friendlier on X) before posting.

---

## 1. X reply (single post, 274 characters)

```
animacy — the open interaction layer for expressive robots: human motion in, any robot out, one ROBOT.md per body. Runs on a real Reachy Mini; writes the Lamp's own /servo/upload CSV.
Video: <VIDEO_URL>
Demo: hcoder10.github.io/animacy/web/
Repo: github.com/Hcoder10/animacy
```

Character counts here are literal, counting `<VIDEO_URL>` as the 11 characters
it is. X shortens every URL to 23 characters regardless of length, so once the
three links are real this post measures 274 − 11 − 37 − 33 + 69 = **262** on
X's counter: it fits whatever the video URL turns out to be. (Apache-2.0 is
dropped from the post for length; it is on the repo page and in the Reddit
comment.)

### Thread variant (reply to your own post, 4 tweets)

**1/4** (254)
```
animacy — the open interaction layer for expressive robots: human motion in, any robot out, one ROBOT.md per body. Runs on a real Reachy Mini today; writes the Lamp's own /servo/upload CSV. Apache-2.0.
Video: <VIDEO_URL>
Repo: github.com/Hcoder10/animacy
```

**2/4** (274)
```
Verified: Reachy Mini sim-to-real on a physical unit (every axis read back within a few degrees, owner confirmed all 5 directions); browser retargeter = Python to 1e-6; 0 speed-cap violations on every clip; Lamp URDF from the vendor CAD, checked against all 31 vendor clips.
```

**3/4** (275) — the blind judge, graded against the vendors' own clips
```
I built a blind judge: a separate model grades animacy's motion and the vendor's OWN hand-made clips in one sealed reel. On the Lamp we beat their clip on excitement (7v6), tie greeting, lose thinking (6v8). Mean 6.0 vs 6.6. My 8/10 bar: met by neither. All 3 runs published.
```

**4/4** (262)
```
Not verified: no Lamp on hand, so that path is checked against Autonomous's own route code, not hardware. The learned model doesn't beat shuffled audio on beat timing, so retrieval ships as the default. Every number and every gap: hcoder10.github.io/animacy/web/
```

Numbers in 3/4 are the run-3 **held-out** (gate) lines for the shipped
`retrieval` source against the `vendor` column, from
`docs/evidence/grading/20260827_1501_run3.md`: lamp greeting 6.0/6.0,
agreement 5.0/6.0, doubt 6.0/7.0, excitement 7.0/6.0, thinking 6.0/8.0; means
6.0 vs 6.6. **Do not quote the tuning-lines table by mistake** — it reads
thinking 7.0, which is not the gate. **Do not claim the Reachy win** (6.2 vs
5.6): that run's vendor calibration is flagged `BROKEN` by the harness, so
those scores are not trustworthy.

---

## 2. Reddit comment (r/coolgithubprojects, ~440 words)

```
**animacy — the open interaction layer for expressive robots** (Apache-2.0)
Repo: https://github.com/Hcoder10/animacy · Demo in the browser: https://hcoder10.github.io/animacy/web/ · Video: <VIDEO_URL>

Human motion in, any robot's motion out. One canonical human-motion space captured from any video or webcam with MediaPipe, and one `ROBOT.md` per robot: joints, limits, safety ceiling, a linear mapping from those channels. Adding a robot is one file; no retraining. The demo plays the Autonomous Lamp and the Reachy Mini side by side on their real URDFs: vendor clips, retargeted human clips, your webcam live, and a talk mode where the robot speaks (TTS in the page) and moves in sync.

**Verified**
- Reachy Mini on a physical unit: clip → `ROBOT.md` → daemon at 30 Hz; every axis read back within a few degrees; owner confirmed all five directions.
- Browser retargeter equals the Python one to 1e-6; 0 speed-cap violations on every clip.
- Lamp URDF from Autonomous's own CAD, directions checked against their device notes and all 31 vendor recordings; the exporter writes the CSV `/servo/upload` accepts.
- 213 passing tests; two LeRobot v3.0 datasets on the Hub; a 73-clip / 321-minute / 37-speaker licence-verified human corpus.

**Not verified / not done**
- No Lamp on hand: the Lamp path is checked against Autonomous's route code, not hardware.
- The learned speech→motion model predicts unseen speakers ~2 nats below the unigram floor, but does **not** beat its own shuffled-audio control on beat timing on either held-out speaker — so no claim that the motion is speech-timed, and retrieval (real human windows, aligned by construction) ships as the default.
- **The project's own acceptance gate fails, and here is exactly how badly.** I built a blind judge: a separate model watches short rendered clips with no idea what produced them, the clip→origin map is sealed, and the test lines are held out. It grades **the vendors' own hand-authored clips in the same reel**, so there is a reference made by the people who built the robot. On the Lamp, animacy's shipped motion is level with or better than the vendor's clips on 2 of 5 movements (excitement 7 vs 6, greeting 6 vs 6) and behind on 3 (thinking 6 vs 8, doubt 6 vs 7, agreement 5 vs 6) — mean 6.0 vs 6.6. Reachy scored above its vendor clips on average, but the harness flagged that run's calibration as broken, so I'm not claiming it. **The 8/10 bar I set is met by nothing — not animacy, not the vendors' own clips.** All three runs published in full.

It targets four items in Autonomous OS's own `docs/not-built-yet.md` (cross-body moves, community moves for Reachy, recordings under the safety gate, a live policy behind `/policy/run`); see the README.
```

Word count: 442 (excluding the link line). If a shorter comment is wanted, cut
the second and third "Not verified" bullets to one line each — but keep the
gate paragraph: publishing a failing self-imposed bar next to the vendors' own
scores is the most credible thing in the post.

---

## 3. Demo video — shot list

> **As rendered:** `docs/media/animacy_demo.mp4` is 1920×1080 H.264 + AAC,
> **3 min 38 s**, 39.6 MB; the tracked 720p cut is 14.5 MB. The two hosts
> narrate the pipeline, moving with motion generated from their own speech.
> The shot list below is the original 60-second plan
> that cut was built from; the timings are the plan's, not the finished file's.

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
| 48–60 s | **The real Reachy Mini (phone video).** Two beats: it speaks and moves; then it mirrors a public-domain clip playing on the laptop next to it. | Beat 1 (6 s): `animacy say "Hello! I'm animacy, running on a real Reachy Mini." --robot reachy_mini --url http://192.168.1.60:8000 --source retrieval --checkpoint checkpoints/v2a` (laptop speaker audible). Beat 2 (6 s): `animacy mirror --source "data/raw/2013_09_14_President_Obama_s_Weekly_Address.webm" --robot reachy_mini --sink reachy_daemon --url http://192.168.1.60:8000 --preview --start 30 --duration 20` with the `--preview` window visible on the laptop in frame. | Beat 1: *Same pipeline, real robot: `animacy say`, audio and motion from one waveform.* Beat 2: *`animacy mirror`: a public-domain weekly address (White House, 2013) driving the robot live at 30 Hz.* |
| last 3 s | **End card.** | Black card, two lines. | *github.com/Hcoder10/animacy · Apache-2.0. Not yet: no Lamp on hand; the learned model ships behind retrieval. Details in README → Status.* |

Notes for the operator:

- Reachy: power on, wait ~1–2 min for the daemon, then the commands above enable motors and `wake_up` themselves (`animacy.sinks.ReachyDaemonSink.prepare`). Frame the head and antennas; the body yaw is visible on big turns.
- `--source retrieval` needs `checkpoints/v2a/retrieval.json` (present on the dev machine, not in git; it is the `animacy say` default). `--source envelope` needs nothing and is a labelled heuristic — if used, the caption must say "envelope heuristic", not "model".
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
| 213 passing tests | `tests/` — `python -m pytest -q` on a clean checkout: 224 collected, **213 passed, 5 skipped** (2026-08-27). `tests/test_lamp_hal.py` needs a live Autonomous OS HAL and skips without one; one of its assertions fails when a HAL *is* present — see `README.md` § Status |
| 73-clip / 321-minute / 37-speaker licence-verified corpus | `data/clips/_index.json` (73 entries with `status: kept`, `valid_s` summing to 19 246 s = 320.8 min, 37 distinct `speaker` values), HF card `squaredcuber/animacy-human-motion`, `docs/HARVEST.md` |
| two LeRobot v3.0 datasets on the Hub validated with the real loader | `docs/LEROBOT.md` (repo ids, validator), `scripts/export_lerobot.py`, `tests/test_lerobot_export.py` |
| no Lamp on hand; Lamp path checked against route code, not hardware | `robots/lamp/ROBOT.md` ("verified in simulation only"), `animacy/sinks.py` (`AutonomousHalSink` docstring: UNVERIFIED), `README.md` § Status |
| the learned model beats its floors on one held-out speaker, not the other; retrieval is the default | `docs/RESULTS.md` § Audio → motion (kende vs obama_2015 columns; "Verdict"), `web/models/model.json` (`default_backend: retrieval`) |
| the blind-grader acceptance gate **fails**; on the Lamp animacy is level-or-better on 2 of 5 movements against the vendor's own clips and behind on 3 (mean 6.0 vs 6.6) | `docs/evidence/grading/20260827_1501_run3.md` § "lamp: held-out lines" (retrieval vs vendor columns: 6.0/6.0, 5.0/6.0, 6.0/7.0, **7.0/6.0**, 6.0/8.0) and the gate table (lamp min 5.0 FAIL, reachy_mini min 5.0 FAIL), earlier runs `20260826_2320.md` and `20260827_0301_run2.md`, spec `docs/GRADING.md`, gate `animacy/grade/run.py` |
| the Reachy vendor-comparison is **not** claimed | same file: "Calibration (vendor clips): mean overall **5.6** over 5 clips (minimum 6.0): BROKEN - vendor clips average below the minimum: the rendering or the rubric is broken, candidate scores are not trustworthy" |
| the learned model does not beat its shuffled-audio control on beat timing on either held-out speaker | `docs/RESULTS.md` § v2a (margins −0.06 obama / −0.01 kende against the required +0.05), `web/models/model.json` (`default_backend: retrieval`) |
| targets four items in Autonomous OS's `docs/not-built-yet.md` | `README.md` § For Autonomous OS (each ask quoted with the command), `docs/AUTONOMOUS_OS.md` |
| Apache-2.0 | `LICENSE`, `pyproject.toml` (`license = Apache-2.0`), `README.md` § License and attributions |
| public-domain clip used in the mirror shot | `data/raw/sources.json` (license record for the 2013-09-14 weekly address), HF dataset card `squaredcuber/animacy-human-motion` |
| ~240 fps on the dev laptop | `docs/RESULTS.md` § Browser (240 fps on an RTX 5080 laptop), `web/dev/fps.py` |
