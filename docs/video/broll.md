# B-roll for the demo film

Footage for `docs/video/script.md`. Every clip is **real**: real stdout from a
command that actually ran on this machine, the real web viewer, the real
dataset pages, the real robot's own read-back. Nothing is a mock-up, a
re-creation or a motion graphic.

- **Format** — 1920x1080, 30 fps, MP4/h264, **silent** (the edit adds narration).
- **Where** — `data/video/broll/<section>_<slug>.mp4` (gitignored; `data/video/` is in `.gitignore`).
- **Index** — `data/video/broll/manifest.json`: for every clip, the section it
  serves, what it shows, its duration and the exact command or URL that produced it.

Read the manifest first; this page explains how the footage is made and what
the editor should know about each kind of shot.

## Running it

Python: `C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe`. From the repo root:

| command | shots |
|---|---|
| `python scripts/video/broll_terminal.py` | every terminal + document shot (`--shots` to pick) |
| `python scripts/video/broll_capture.py` | §2 `animacy capture --preview`, screen-recorded |
| `python scripts/video/broll_viewer.py` | §2/§4/§5 the web viewer |
| `python scripts/video/broll_web.py` | §7/§9 Hugging Face, the live site, GitHub |
| `python scripts/video/broll_grading.py` | §8 a slice of a real grading reel |
| `python scripts/video/broll_waveform.py` | §5 speech waveform under the motion it produced |
| `python scripts/video/broll_hardware.py` | §6 registers the supplied phone footage and cuts an excerpt |

Each script re-registers only the clips it makes, so any one can be re-run
without disturbing the rest. Keep to one at a time — they all record.

## How each kind of shot stays real

**Terminal shots** (`broll_terminal.py` → `web/dev/broll/term.html`).
The command runs for real first; its stdout and exit code are captured
verbatim, then replayed in a styled monospace page that only controls the
*pace* at which those bytes appear. The terminal even waits roughly as long as
the real command took before its output starts. Where a shot shows `animacy …`,
the venv's own `animacy.exe` console script is what ran; POSIX one-liners
(`head`, `wc`, `grep`, `ls`) run in Git Bash from the repo root. The type size
is fitted so the widest captured line fills the frame without wrapping — that
is the only presentational choice.

**Document shots** (`web/dev/broll/doc.html`). A real file's text, scrolled at
reading pace or shrunk to fit and held still. The highlighter only wraps spans
around characters that are already there; it never rewrites a line. Each shot
names the file and the line range it shows.

**Viewer shots** (`broll_viewer.py`). The page is parked
(`animacy.setCapture(true)`) and advanced a frame at a time
(`animacy.stepFrame(1/30)`), so frame *i* is exactly *i*/30 s of clip time
however slow the renderer is. Every joint angle comes from the real retargeter
reading the real `ROBOT.md`. Clip time is exact; wall-clock render time is not,
which matters for the Talk shot (see below).

**Screen recording** (`broll_capture.py`). `animacy capture --preview` runs over
a licensed clip and its OpenCV window is grabbed with `ffmpeg -f gdigrab`.
OpenCV's window is DPI-unaware, so on a scaled display the drawn image sits in
the top-left of the reported client area; the script crops to the source
video's own frame size to remove the surrounding desktop.

**Browser shots** (`broll_web.py`). Playwright's own video recorder, in real
time, with the pointer moved in steps and the page scrolled a little at a time.
The pages are live; the only thing the script decides is where the cursor goes.

## Notes the edit should know

- **§2 channel bars** — the readout is a wide band (about 5.6:1), so the frame
  is that band scaled to full width and centred on the film's background. Crop
  or overlay it as you prefer.
- **§5 Talk** — the line is typed, the button is pressed, and the page's own
  pipeline runs: Kokoro-82M in the browser → speech features → motion
  retrieval → both robots. Kokoro runs on wasm here (Playwright's Chromium
  reports a WebGPU adapter it cannot open a device on), which takes about 12 s
  of wall time to synthesise; because the page is stepped a frame at a time,
  that wait is not in the clip. The motion and its clock are real; the render
  wall time is not the clip's time.
- **§5 waveform** — both traces come from the same graded clip: the audio is
  the TTS the grader synthesised, the joint curves are what `animacy` generated
  from it. The reel part starts with a 1 s title card, which is trimmed so the
  two share a zero. The speech ends before the motion does — that tail is the
  settle, and it is real.
- **§6 read-back** — the sim-to-real replay reports honestly that the head and
  body axes track within a few degrees while the **right antenna reads 61.4° of
  90° commanded** at the brow peak. That outlier is in the published evidence
  table too. If the narration's "every axis read back within a couple of
  degrees" is meant to cover antennas, it needs a tweak.
- **§6 live poll** — the ambient loop rests between clips, so the poll runs 45 s
  and prints the busiest 4.5 s of it. Every row is a real sample at its real
  timestamp and the on-screen header says exactly that.
- **§8 grading reel** — the "Clip N" cards stay in frame on purpose: the card
  gives the spoken line and nothing else, no robot, no source, no method. That
  anonymity is the test. Window offsets come from the run's own sealed manifest
  and the reel builder's `CARD_SECONDS`/`GAP_SECONDS`, so the cut lands on a
  card boundary.
- **§8 score table** — the shot shows animacy **above** the vendor on
  excitement (8.0 vs 6.0) and **below** on thinking (4.0 vs 8.0). That is the
  "level on some movements and below on others" line, on screen, unedited.
- Two clips are not 16:9 in origin and are pillar/letterboxed on the film's own
  background (`#0e1117`): the grading reel (512x512) and the channel band.
- **Footage this pipeline did not make.**
  `real_reachy_desk_16x9.mp4` and `real_reachy_desk_portrait.mp4` are phone
  footage of the physical Reachy Mini, filmed by a person and placed in
  `data/video/broll/` for the edit. They are in the manifest so the edit knows
  they exist, are kept **untouched** (33.6 s each, so outside the 6-15 s spec
  that applies to generated clips), and are marked as supplied. Their
  provenance is only what the file shows — do not present them as evidence of a
  specific run without checking. `s6_reachy_hardware.mp4` is a 12 s excerpt of
  the 16:9 take, conformed to the film's format; its window was chosen by
  frame-difference energy rather than by eye.

## Artifacts written outside `data/video/`

`out/lamp_obama.csv` and `out/data_report.txt` are produced by the shots that
show them, and `out/broll_capture_preview/` is the clip the preview run writes.
`out/` is untracked scratch.
